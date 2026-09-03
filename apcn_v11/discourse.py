from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json

from apcn_v10.semantic import EntityRef, SemanticNode


@dataclass
class DiscourseEntity:
    ref: EntityRef
    salience: float = 0.0
    mentions: int = 0
    last_turn: int = 0
    last_role: str = "unknown"

    @property
    def key(self) -> str:
        return self.ref.key()

    def to_dict(self) -> Dict[str, object]:
        return {
            "ref": self.ref.to_dict(),
            "salience": self.salience,
            "mentions": self.mentions,
            "last_turn": self.last_turn,
            "last_role": self.last_role,
        }

    @classmethod
    def from_dict(cls, row: Dict[str, object]) -> "DiscourseEntity":
        r = row["ref"]
        return cls(
            EntityRef(str(r["color"]), str(r["shape"]), int(r.get("instance", 0))),
            float(row.get("salience", 0.0)),
            int(row.get("mentions", 0)),
            int(row.get("last_turn", 0)),
            str(row.get("last_role", "unknown")),
        )


class DiscourseEntityRegistry:
    """Small persistent discourse-state memory.

    This is deliberately separate from long-term semantic memory. It tracks
    entities active in the current discourse, their stable instance identities,
    recency and salience. It does not retain complete utterance history.
    """

    VERSION = "APCN-V0.11-DISCOURSE-REGISTRY"

    def __init__(self, *, salience_decay: float = 0.82, max_entities: int = 32):
        self.salience_decay = float(salience_decay)
        self.max_entities = int(max_entities)
        self.turn = 0
        self.entities: Dict[str, DiscourseEntity] = {}
        self.focus_key: Optional[str] = None
        self.next_instance = 0

    def reset(self) -> None:
        self.turn = 0
        self.entities.clear()
        self.focus_key = None
        self.next_instance = 0

    @property
    def focus(self) -> Optional[EntityRef]:
        row = self.entities.get(self.focus_key or "")
        return None if row is None else row.ref

    def _decay(self) -> None:
        for row in self.entities.values():
            row.salience *= self.salience_decay

    def _reserve_instance(self, instance: int) -> None:
        self.next_instance = max(self.next_instance, int(instance) + 1)

    def register(self, ref: EntityRef, *, role: str = "mention", focus: bool = False,
                 salience_boost: float = 1.0) -> EntityRef:
        self._reserve_instance(ref.instance)
        key = ref.key()
        row = self.entities.get(key)
        if row is None:
            row = DiscourseEntity(ref)
            self.entities[key] = row
        row.mentions += 1
        row.last_turn = self.turn
        row.last_role = str(role)
        row.salience = max(row.salience, 0.0) + float(salience_boost)
        if focus:
            self.focus_key = key
        self._prune()
        return row.ref

    def new_entity(self, color: str, shape: str, *, role: str = "mention",
                   focus: bool = False) -> EntityRef:
        ref = EntityRef(str(color), str(shape), self.next_instance)
        self.next_instance += 1
        return self.register(ref, role=role, focus=focus)

    def matching(self, color: str, shape: str) -> List[DiscourseEntity]:
        rows = [r for r in self.entities.values()
                if r.ref.color == color and r.ref.shape == shape]
        rows.sort(key=lambda r: (r.salience, r.last_turn, r.mentions), reverse=True)
        return rows

    def resolve_description(self, color: str, shape: str, *, create: bool = True,
                            prefer_existing: bool = True, role: str = "mention") -> Optional[EntityRef]:
        rows = self.matching(str(color), str(shape))
        if rows and prefer_existing:
            ref = rows[0].ref
            self.register(ref, role=role, salience_boost=.55)
            return ref
        if not create:
            return None
        return self.new_entity(str(color), str(shape), role=role)

    def resolve_reference(self, *, role: str = "subject") -> Optional[EntityRef]:
        if self.focus is not None:
            ref = self.focus
            self.register(ref, role=role, focus=True, salience_boost=.75)
            return ref
        if not self.entities:
            return None
        rows = sorted(self.entities.values(),
                      key=lambda r: (r.salience, r.last_turn, r.mentions), reverse=True)
        ref = rows[0].ref
        self.register(ref, role=role, focus=True, salience_boost=.75)
        return ref

    def ingest(self, program: Optional[SemanticNode]) -> None:
        """Update discourse state from APCN's own parsed program.

        Evaluation code should call this with the learner's prediction, never the
        teacher's expected program, so context cannot leak answer metadata.
        """
        if program is None:
            return
        self.turn += 1
        self._decay()
        atoms = [n for n in program.walk() if n.op == "RELATION"]
        for atom in atoms:
            if atom.subject is not None:
                self.register(atom.subject, role="subject", focus=True, salience_boost=1.2)
            if atom.object is not None:
                self.register(atom.object, role="object", salience_boost=.75)

    def seed(self, refs: Iterable[EntityRef], *, focus: Optional[EntityRef] = None) -> None:
        self.reset()
        for ref in refs:
            self.register(ref, role="seed", focus=False, salience_boost=.5)
        if focus is not None:
            self.register(focus, role="focus", focus=True, salience_boost=1.0)

    def _prune(self) -> None:
        if len(self.entities) <= self.max_entities:
            return
        protected = {self.focus_key} if self.focus_key else set()
        rows = sorted(self.entities.values(), key=lambda r: (r.salience, r.last_turn, r.mentions))
        for row in rows:
            if len(self.entities) <= self.max_entities:
                break
            if row.key in protected:
                continue
            self.entities.pop(row.key, None)

    def summary(self) -> Dict[str, object]:
        rows = sorted(self.entities.values(),
                      key=lambda r: (r.salience, r.last_turn, r.mentions), reverse=True)
        return {
            "turn": self.turn,
            "entity_count": len(rows),
            "focus": None if self.focus is None else self.focus.to_dict(),
            "next_instance": self.next_instance,
            "entities": [r.to_dict() for r in rows],
        }

    def save(self, path: str | Path) -> None:
        data = {
            "version": self.VERSION,
            "salience_decay": self.salience_decay,
            "max_entities": self.max_entities,
            "turn": self.turn,
            "focus_key": self.focus_key,
            "next_instance": self.next_instance,
            "entities": [r.to_dict() for r in self.entities.values()],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DiscourseEntityRegistry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(salience_decay=float(data.get("salience_decay", .82)),
                  max_entities=int(data.get("max_entities", 32)))
        obj.turn = int(data.get("turn", 0))
        obj.focus_key = data.get("focus_key")
        obj.next_instance = int(data.get("next_instance", 0))
        for raw in data.get("entities", []):
            row = DiscourseEntity.from_dict(raw)
            obj.entities[row.key] = row
            obj._reserve_instance(row.ref.instance)
        return obj
