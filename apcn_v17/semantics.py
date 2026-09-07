from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple
import re


def normalize_term(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text).strip().lower())
    return text.strip(" .,:;!?\t\n")


@dataclass(frozen=True)
class SemanticClause:
    """Explicit compositional semantic structure used by APCN V0.17.

    `op` identifies a semantic operator. Atomic clauses keep their arguments in
    `slots`; higher-order clauses keep nested propositions in `children`.
    Language may map to/from this object, but the object itself is not English.
    """

    op: str
    slots: Tuple[Tuple[str, str], ...] = ()
    children: Tuple["SemanticClause", ...] = ()
    tense: str = "atemporal"

    @classmethod
    def make(
        cls,
        op: str,
        /,
        *,
        children: Sequence["SemanticClause"] = (),
        tense: str = "atemporal",
        **slots: object,
    ) -> "SemanticClause":
        pairs = tuple(sorted((str(k), normalize_term(str(v))) for k, v in slots.items() if str(v).strip()))
        return cls(str(op).upper(), pairs, tuple(children), normalize_term(tense) or "atemporal")

    @classmethod
    def atom(
        cls,
        subject: str,
        predicate: str,
        object: Optional[str] = None,
        *,
        tense: str = "present",
    ) -> "SemanticClause":
        slots: Dict[str, object] = {"subject": subject, "predicate": predicate}
        if object is not None:
            slots["object"] = object
        return cls.make("ATOM", tense=tense, **slots)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        for k, v in self.slots:
            if k == key:
                return v
        return default

    def slot_dict(self) -> Dict[str, str]:
        return dict(self.slots)

    def to_dict(self) -> Dict[str, object]:
        return {
            "op": self.op,
            "slots": dict(self.slots),
            "children": [child.to_dict() for child in self.children],
            "tense": self.tense,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SemanticClause":
        slots_raw = data.get("slots", {})
        if isinstance(slots_raw, Mapping):
            slots = tuple(sorted((str(k), str(v)) for k, v in slots_raw.items()))
        else:
            slots = tuple((str(k), str(v)) for k, v in slots_raw)  # type: ignore[arg-type]
        children = tuple(cls.from_dict(row) for row in data.get("children", []))  # type: ignore[arg-type]
        return cls(str(data.get("op", "ATOM")).upper(), slots, children, str(data.get("tense", "atemporal")))

    def mentions(self) -> Tuple[str, ...]:
        out = []
        for key in ("subject", "object", "kind", "category"):
            value = self.get(key)
            if value and not value.startswith("$"):
                out.append(value)
        for child in self.children:
            out.extend(child.mentions())
        return tuple(dict.fromkeys(out))

    def replace_entity(self, old: str, new: str) -> "SemanticClause":
        old_n, new_n = normalize_term(old), normalize_term(new)
        slots = {k: (new_n if normalize_term(v) == old_n else v) for k, v in self.slots}
        return SemanticClause.make(
            self.op,
            children=tuple(child.replace_entity(old_n, new_n) for child in self.children),
            tense=self.tense,
            **slots,
        )

    def structural_size(self) -> int:
        return 1 + sum(child.structural_size() for child in self.children)


ATOMIC_OPS = {"ATOM", "IS_A", "PROPERTY", "RELATION", "EVENT"}
HIGHER_OPS = {"NOT", "CAUSE", "IF", "BEFORE", "AFTER", "AND", "FORALL_ISA", "EXISTS_PROPERTY"}


def semantic_equal(a: SemanticClause, b: SemanticClause) -> bool:
    return a == b


def walk(clause: SemanticClause) -> Iterable[SemanticClause]:
    yield clause
    for child in clause.children:
        yield from walk(child)
