from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from math import sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import re

from apcn_v17.semantics import SemanticClause, normalize_term, walk


@dataclass
class ConceptBindingV18:
    alias: str
    target: str
    source: str = "user"
    support: int = 1
    first_index: int = 0
    last_index: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "alias": self.alias,
            "target": self.target,
            "source": self.source,
            "support": self.support,
            "first_index": self.first_index,
            "last_index": self.last_index,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ConceptBindingV18":
        return cls(
            alias=normalize_term(str(data.get("alias", ""))),
            target=normalize_term(str(data.get("target", ""))),
            source=str(data.get("source", "user")),
            support=max(1, int(data.get("support", 1))),
            first_index=int(data.get("first_index", 0)),
            last_index=int(data.get("last_index", 0)),
        )


class ConceptBindingMemoryV18:
    """Explicit lexical-symbol -> canonical-concept identity memory.

    Bindings are applied before semantic reasoning. They do not copy facts and
    cannot answer truth questions themselves.
    """

    VERSION = "APCN-V0.18-CONCEPT-BINDING"

    def __init__(self, max_bindings: int = 4096):
        self.max_bindings = int(max_bindings)
        self.bindings: Dict[str, ConceptBindingV18] = {}
        self.observations = 0
        self.raw_sentences_retained = 0

    def resolve(self, term: str) -> str:
        current = normalize_term(term)
        seen = set()
        while current in self.bindings and current not in seen:
            seen.add(current)
            current = normalize_term(self.bindings[current].target)
        return current

    def bind(self, alias: str, target: str, *, source: str = "user", weight: int = 1) -> ConceptBindingV18:
        alias_n = normalize_term(alias)
        target_n = self.resolve(target)
        if not alias_n or not target_n:
            raise ValueError("alias and target must be non-empty")
        if alias_n == target_n:
            raise ValueError("alias and target resolve to the same concept")

        # Follow the proposed target chain. If it reaches alias, the update is cyclic.
        probe = target_n
        seen = {alias_n}
        while probe in self.bindings:
            if probe in seen:
                raise ValueError("concept binding would create a cycle")
            seen.add(probe)
            probe = normalize_term(self.bindings[probe].target)

        self.observations += 1
        rec = self.bindings.get(alias_n)
        if rec is None:
            rec = ConceptBindingV18(
                alias=alias_n,
                target=target_n,
                source=source,
                support=max(1, int(weight)),
                first_index=self.observations,
                last_index=self.observations,
            )
            self.bindings[alias_n] = rec
        else:
            rec.target = target_n
            rec.source = source or rec.source
            rec.support += max(1, int(weight))
            rec.last_index = self.observations
        self._prune()
        return rec

    def _prune(self) -> None:
        if len(self.bindings) <= self.max_bindings:
            return
        rows = sorted(self.bindings.values(), key=lambda r: (r.support, r.last_index, r.alias))
        for rec in rows[: len(self.bindings) - self.max_bindings]:
            self.bindings.pop(rec.alias, None)

    def canonicalize_clause(self, clause: SemanticClause) -> SemanticClause:
        slots = clause.slot_dict()
        for key in ("subject", "object", "kind", "category"):
            if key in slots:
                slots[key] = self.resolve(slots[key])
        return SemanticClause.make(
            clause.op,
            children=tuple(self.canonicalize_clause(c) for c in clause.children),
            tense=clause.tense,
            **slots,
        )

    def canonicalize_surface(self, text: str) -> Dict[str, object]:
        """Apply exact taught aliases only; approximate terms are never committed."""
        surface = str(text)
        replacements: List[Dict[str, str]] = []
        for alias in sorted(self.bindings, key=lambda x: (-len(x), x)):
            target = self.resolve(alias)
            if not alias or alias == target:
                continue
            pattern = re.compile(r"(?<![\w'-])" + re.escape(alias) + r"(?![\w'-])", re.I)
            if pattern.search(surface):
                surface = pattern.sub(target, surface)
                replacements.append({"alias": alias, "canonical": target})
        return {
            "surface": surface,
            "changed": bool(replacements),
            "replacements": replacements,
        }

    def summary(self, limit: int = 16) -> Dict[str, object]:
        rows = sorted(self.bindings.values(), key=lambda r: (r.support, r.last_index), reverse=True)
        return {
            "version": self.VERSION,
            "observations": self.observations,
            "bindings": len(self.bindings),
            "max_bindings": self.max_bindings,
            "strongest": [r.to_dict() for r in rows[:limit]],
            "truth_authority": False,
            "raw_sentences_retained": 0,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "max_bindings": self.max_bindings,
            "observations": self.observations,
            "bindings": [r.to_dict() for r in self.bindings.values()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ConceptBindingMemoryV18":
        obj = cls(int(data.get("max_bindings", 4096)))
        obj.observations = int(data.get("observations", 0))
        for raw in data.get("bindings", []):
            rec = ConceptBindingV18.from_dict(raw)  # type: ignore[arg-type]
            if rec.alias and rec.target:
                obj.bindings[rec.alias] = rec
        obj._prune()
        return obj

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ConceptBindingMemoryV18":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class PrototypeRecordV18:
    concept: str
    vector: List[float]
    updates: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {"concept": self.concept, "vector": self.vector, "updates": self.updates}


class OnlinePrototypeMemoryV18:
    """Bounded distributed concept state learned from APCN semantic events.

    This is intentionally not a truth store and not a pretrained embedding model.
    Signed feature hashing turns explicit semantic roles/relations into an online
    vector prototype that can later support similarity and hypothesis ranking.
    """

    VERSION = "APCN-V0.18-ONLINE-PROTOTYPES"

    def __init__(self, dimensions: int = 256, max_concepts: int = 8192):
        if int(dimensions) < 32:
            raise ValueError("dimensions must be >= 32")
        self.dimensions = int(dimensions)
        self.max_concepts = int(max_concepts)
        self.records: Dict[str, PrototypeRecordV18] = {}
        self.observations = 0
        self.raw_sentences_retained = 0

    def _index_sign(self, feature: str) -> Tuple[int, float]:
        digest = blake2b(feature.encode("utf-8"), digest_size=16).digest()
        return int.from_bytes(digest[:8], "big") % self.dimensions, (1.0 if digest[8] & 1 else -1.0)

    def _record(self, concept: str) -> PrototypeRecordV18:
        key = normalize_term(concept)
        rec = self.records.get(key)
        if rec is None:
            rec = PrototypeRecordV18(key, [0.0] * self.dimensions, 0)
            self.records[key] = rec
        return rec

    def _add(self, concept: str, features: Iterable[str], weight: float = 1.0) -> None:
        rec = self._record(concept)
        touched = False
        for feature in features:
            index, sign = self._index_sign(feature)
            rec.vector[index] += sign * float(weight)
            touched = True
        if touched:
            rec.updates += 1

    def observe_clause(self, clause: SemanticClause, bindings: Optional[ConceptBindingMemoryV18] = None) -> None:
        resolver = bindings.resolve if bindings is not None else normalize_term
        self.observations += 1
        for node in walk(clause):
            mentions: List[Tuple[str, str]] = []
            for role in ("subject", "object", "kind", "category"):
                value = node.get(role)
                if value and not value.startswith("$"):
                    mentions.append((role, resolver(value)))
            for role, concept in mentions:
                features = [f"op={node.op}", f"role={role}", f"tense={node.tense}"]
                for other_role, other in mentions:
                    if (other_role, other) != (role, concept):
                        features.append(f"peer:{other_role}={other}")
                for key in ("predicate", "property", "relation", "event"):
                    value = node.get(key)
                    if value:
                        features.append(f"{key}={normalize_term(value)}")
                self._add(concept, features)
        self._prune()

    def observe_definition(
        self,
        concept: str,
        dependencies: Iterable[str],
        *,
        kind: str = "defined",
        bindings: Optional[ConceptBindingMemoryV18] = None,
    ) -> None:
        resolver = bindings.resolve if bindings is not None else normalize_term
        key = resolver(concept)
        deps = sorted({resolver(d) for d in dependencies if normalize_term(d)})
        features = [f"concept_kind={normalize_term(kind) or 'defined'}"] + [f"depends_on={d}" for d in deps]
        self._add(key, features)
        for dep in deps:
            self._add(dep, [f"dependency_of={key}"], weight=.5)
        self.observations += 1
        self._prune()

    def bind_alias(self, alias: str, target: str, bindings: ConceptBindingMemoryV18) -> None:
        target_n = bindings.resolve(target)
        self._add(target_n, [f"alias={normalize_term(alias)}", "relation=concept_identity"])
        self.observations += 1
        self._prune()

    def vector_for(self, concept: str, bindings: Optional[ConceptBindingMemoryV18] = None) -> Optional[Tuple[float, ...]]:
        key = bindings.resolve(concept) if bindings is not None else normalize_term(concept)
        rec = self.records.get(key)
        return None if rec is None else tuple(rec.vector)

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sqrt(sum(x * x for x in a))
        nb = sqrt(sum(y * y for y in b))
        return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)

    def similarity(self, a: str, b: str, bindings: Optional[ConceptBindingMemoryV18] = None) -> float:
        va, vb = self.vector_for(a, bindings), self.vector_for(b, bindings)
        return 0.0 if va is None or vb is None else self._cosine(va, vb)

    def nearest(self, concept: str, *, bindings: Optional[ConceptBindingMemoryV18] = None, limit: int = 8) -> List[Dict[str, object]]:
        key = bindings.resolve(concept) if bindings is not None else normalize_term(concept)
        query = self.vector_for(key, bindings)
        if query is None:
            return []
        rows = [
            {"concept": other, "similarity": self._cosine(query, rec.vector), "updates": rec.updates}
            for other, rec in self.records.items() if other != key
        ]
        rows.sort(key=lambda r: float(r["similarity"]), reverse=True)
        return rows[: max(1, int(limit))]

    def _prune(self) -> None:
        if len(self.records) <= self.max_concepts:
            return
        rows = sorted(self.records.values(), key=lambda r: (r.updates, r.concept))
        for rec in rows[: len(self.records) - self.max_concepts]:
            self.records.pop(rec.concept, None)

    def summary(self, limit: int = 16) -> Dict[str, object]:
        rows = sorted(self.records.values(), key=lambda r: (r.updates, r.concept), reverse=True)
        return {
            "version": self.VERSION,
            "dimensions": self.dimensions,
            "observations": self.observations,
            "concepts": len(self.records),
            "max_concepts": self.max_concepts,
            "strongest": [{"concept": r.concept, "updates": r.updates} for r in rows[:limit]],
            "truth_authority": False,
            "pretrained_embeddings": False,
            "external_model": False,
            "raw_sentences_retained": 0,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "dimensions": self.dimensions,
            "max_concepts": self.max_concepts,
            "observations": self.observations,
            "records": [r.to_dict() for r in self.records.values()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "OnlinePrototypeMemoryV18":
        obj = cls(int(data.get("dimensions", 256)), int(data.get("max_concepts", 8192)))
        obj.observations = int(data.get("observations", 0))
        for raw in data.get("records", []):
            concept = normalize_term(str(raw.get("concept", "")))  # type: ignore[union-attr]
            vector = [float(x) for x in raw.get("vector", [])]  # type: ignore[union-attr]
            if concept and len(vector) == obj.dimensions:
                obj.records[concept] = PrototypeRecordV18(concept, vector, int(raw.get("updates", 0)))  # type: ignore[union-attr]
        obj._prune()
        return obj

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "OnlinePrototypeMemoryV18":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
