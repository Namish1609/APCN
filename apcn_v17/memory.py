from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import json

from .semantics import SemanticClause, normalize_term


@dataclass
class SemanticMemoryRecord:
    clause: SemanticClause
    source: str = "user"
    support: int = 1
    first_index: int = 0
    last_index: int = 0

    def key(self) -> str:
        return json.dumps(self.clause.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> Dict[str, object]:
        return {
            "clause": self.clause.to_dict(),
            "source": self.source,
            "support": self.support,
            "first_index": self.first_index,
            "last_index": self.last_index,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SemanticMemoryRecord":
        return cls(
            clause=SemanticClause.from_dict(data["clause"]),  # type: ignore[arg-type]
            source=str(data.get("source", "user")),
            support=int(data.get("support", 1)),
            first_index=int(data.get("first_index", 0)),
            last_index=int(data.get("last_index", 0)),
        )


class StructuredSemanticMemoryV17:
    """Bounded explicit proposition/rule memory.

    The memory stores semantic clauses, never raw natural-language sentences.
    It supports a small transparent inference set for V0.17: exact lookup,
    universal category inheritance, conditional consequence lookup, causal
    explanation lookup, and temporal-order lookup.
    """

    VERSION = "APCN-V0.17-STRUCTURED-SEMANTIC-MEMORY"

    def __init__(self, max_records: int = 4096):
        self.max_records = int(max_records)
        self.records: Dict[str, SemanticMemoryRecord] = {}
        self.observations = 0
        self.raw_sentences_retained = 0

    @staticmethod
    def _key(clause: SemanticClause) -> str:
        return json.dumps(clause.to_dict(), sort_keys=True, separators=(",", ":"))

    def teach(self, clause: SemanticClause, *, source: str = "user", weight: int = 1) -> SemanticMemoryRecord:
        self.observations += 1
        key = self._key(clause)
        rec = self.records.get(key)
        if rec is None:
            rec = SemanticMemoryRecord(clause, source, max(1, int(weight)), self.observations, self.observations)
            self.records[key] = rec
        else:
            rec.support += max(1, int(weight))
            rec.last_index = self.observations
            rec.source = source or rec.source
        self._prune()
        return rec

    def _prune(self) -> None:
        if len(self.records) <= self.max_records:
            return
        ranked = sorted(self.records.items(), key=lambda row: (row[1].support, row[1].last_index, row[0]))
        for key, _ in ranked[: len(self.records) - self.max_records]:
            self.records.pop(key, None)

    def contains(self, clause: SemanticClause) -> bool:
        return self._key(clause) in self.records

    def record_for(self, clause: SemanticClause) -> Optional[SemanticMemoryRecord]:
        return self.records.get(self._key(clause))

    def _all(self, op: Optional[str] = None) -> List[SemanticMemoryRecord]:
        rows = list(self.records.values())
        if op is not None:
            rows = [row for row in rows if row.clause.op == op]
        return sorted(rows, key=lambda row: (row.support, row.last_index), reverse=True)

    def infer_truth(self, clause: SemanticClause) -> Dict[str, object]:
        exact = self.record_for(clause)
        if exact is not None:
            return {"known": True, "truth": True, "mode": "explicit", "support": exact.support, "source": exact.source, "evidence": [clause.to_dict()]}

        if clause.op == "IS_A":
            subject = clause.get("subject")
            category = clause.get("category")
            if subject and category:
                for fact in self._all("IS_A"):
                    if fact.clause.get("subject") != subject:
                        continue
                    kind = fact.clause.get("category")
                    if not kind:
                        continue
                    for rule in self._all("FORALL_ISA"):
                        if rule.clause.get("kind") == kind and rule.clause.get("category") == category:
                            return {
                                "known": True,
                                "truth": True,
                                "mode": "universal_inference",
                                "support": min(fact.support, rule.support),
                                "source": "inference",
                                "evidence": [fact.clause.to_dict(), rule.clause.to_dict()],
                            }

        negated = SemanticClause.make("NOT", children=(clause,))
        neg = self.record_for(negated)
        if neg is not None:
            return {"known": True, "truth": False, "mode": "explicit_negation", "support": neg.support, "source": neg.source, "evidence": [negated.to_dict()]}

        if clause.op == "NOT" and clause.children:
            inner = self.record_for(clause.children[0])
            if inner is not None:
                return {"known": True, "truth": False, "mode": "explicit_positive_conflict", "support": inner.support, "source": inner.source, "evidence": [inner.clause.to_dict()]}

        return {"known": False, "truth": None, "mode": "unknown", "support": 0, "source": None, "evidence": []}

    def consequences_for(self, condition: SemanticClause) -> List[Dict[str, object]]:
        out = []
        for rec in self._all("IF"):
            if len(rec.clause.children) != 2 or rec.clause.children[0] != condition:
                continue
            out.append({"consequence": rec.clause.children[1], "rule": rec.clause, "support": rec.support, "source": rec.source})
        return out

    def causes_for(self, effect: SemanticClause) -> List[Dict[str, object]]:
        out = []
        for rec in self._all("CAUSE"):
            if len(rec.clause.children) != 2 or rec.clause.children[1] != effect:
                continue
            out.append({"cause": rec.clause.children[0], "relation": rec.clause, "support": rec.support, "source": rec.source})
        return out

    def before(self, second: SemanticClause) -> List[Dict[str, object]]:
        out = []
        for rec in self._all("BEFORE"):
            if len(rec.clause.children) == 2 and rec.clause.children[1] == second:
                out.append({"first": rec.clause.children[0], "relation": rec.clause, "support": rec.support, "source": rec.source})
        for rec in self._all("AFTER"):
            if len(rec.clause.children) == 2 and rec.clause.children[0] == second:
                out.append({"first": rec.clause.children[1], "relation": rec.clause, "support": rec.support, "source": rec.source})
        return out

    def summary(self, limit: int = 16) -> Dict[str, object]:
        by_op: Dict[str, int] = {}
        for rec in self.records.values():
            by_op[rec.clause.op] = by_op.get(rec.clause.op, 0) + 1
        strongest = self._all()[:limit]
        return {
            "version": self.VERSION,
            "observations": self.observations,
            "records": len(self.records),
            "max_records": self.max_records,
            "by_op": by_op,
            "strongest": [row.to_dict() for row in strongest],
            "raw_sentences_retained": 0,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "max_records": self.max_records,
            "observations": self.observations,
            "records": [row.to_dict() for row in self.records.values()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "StructuredSemanticMemoryV17":
        obj = cls(int(data.get("max_records", 4096)))
        obj.observations = int(data.get("observations", 0))
        for row in data.get("records", []):
            rec = SemanticMemoryRecord.from_dict(row)  # type: ignore[arg-type]
            obj.records[rec.key()] = rec
        obj._prune()
        return obj

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StructuredSemanticMemoryV17":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class DiscourseEntityV17:
    name: str
    salience: float = 1.0
    mentions: int = 1
    last_turn: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "salience": self.salience, "mentions": self.mentions, "last_turn": self.last_turn}


class SemanticDiscourseV17:
    """Bounded semantic discourse focus; raw user utterances are never stored."""

    VERSION = "APCN-V0.17-SEMANTIC-DISCOURSE"
    PRONOUNS = {"it", "this", "that", "this object", "that object", "the same object"}

    def __init__(self, max_entities: int = 64, decay: float = .84):
        self.max_entities = int(max_entities)
        self.decay = float(decay)
        self.entities: Dict[str, DiscourseEntityV17] = {}
        self.focus: Optional[str] = None
        self.turn = 0
        self.last_clause: Optional[SemanticClause] = None

    def resolve(self, term: str) -> str:
        term_n = normalize_term(term)
        if term_n in self.PRONOUNS and self.focus:
            return self.focus
        return term_n

    def _touch(self, name: str) -> None:
        name = normalize_term(name)
        if not name or name.startswith("$"):
            return
        for rec in self.entities.values():
            rec.salience *= self.decay
        rec = self.entities.get(name)
        if rec is None:
            rec = DiscourseEntityV17(name=name, last_turn=self.turn)
            self.entities[name] = rec
        else:
            rec.salience += 1.0
            rec.mentions += 1
            rec.last_turn = self.turn
        self.focus = name
        if len(self.entities) > self.max_entities:
            protected = {self.focus}
            ranked = sorted((r for r in self.entities.values() if r.name not in protected), key=lambda r: (r.salience, r.last_turn))
            for old in ranked[: len(self.entities) - self.max_entities]:
                self.entities.pop(old.name, None)

    def ingest(self, clause: SemanticClause) -> None:
        self.turn += 1
        self.last_clause = clause
        preferred: List[str] = []
        for node in self._walk(clause):
            subject = node.get("subject")
            obj = node.get("object")
            if obj:
                preferred.append(obj)
            if subject:
                preferred.append(subject)
            kind = node.get("kind")
            if kind:
                preferred.append(kind)
        for name in preferred:
            self._touch(name)

    def _walk(self, clause: SemanticClause):
        yield clause
        for child in clause.children:
            yield from self._walk(child)

    def summary(self) -> Dict[str, object]:
        rows = sorted(self.entities.values(), key=lambda r: (r.salience, r.last_turn), reverse=True)
        return {
            "version": self.VERSION,
            "focus": self.focus,
            "turn": self.turn,
            "entities": [r.to_dict() for r in rows],
            "last_clause": None if self.last_clause is None else self.last_clause.to_dict(),
            "raw_chat_transcript_persisted": False,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "max_entities": self.max_entities,
            "decay": self.decay,
            "focus": self.focus,
            "turn": self.turn,
            "entities": [r.to_dict() for r in self.entities.values()],
            "last_clause": None if self.last_clause is None else self.last_clause.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SemanticDiscourseV17":
        obj = cls(int(data.get("max_entities", 64)), float(data.get("decay", .84)))
        obj.focus = data.get("focus") if data.get("focus") is None else str(data.get("focus"))
        obj.turn = int(data.get("turn", 0))
        for row in data.get("entities", []):
            rec = DiscourseEntityV17(str(row["name"]), float(row.get("salience", 1.0)), int(row.get("mentions", 1)), int(row.get("last_turn", 0)))  # type: ignore[index]
            obj.entities[rec.name] = rec
        if data.get("last_clause"):
            obj.last_clause = SemanticClause.from_dict(data["last_clause"])  # type: ignore[arg-type]
        return obj

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SemanticDiscourseV17":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
