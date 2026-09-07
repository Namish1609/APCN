from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re

from apcn_v10.definitions import normalize_name


@dataclass
class AliasRecord:
    alias: str
    target: str
    support: int = 1
    source: str = "user"


class LexicalSemanticMemory:
    """Bounded explicit alias/synonym memory.

    This is deliberately not a word embedding table. It stores inspectable
    symbolic lexical links such as `fluxion -> acceleration`, with support and
    bounded growth. Raw conversations are not retained here.
    """

    VERSION = "APCN-V0.15-LEXICAL-SEMANTIC-MEMORY"

    def __init__(self, max_aliases: int = 8192):
        self.max_aliases = int(max_aliases)
        self.aliases: Dict[str, AliasRecord] = {}
        self.observations = 0

    def teach_alias(self, alias: str, target: str, *, source: str = "user") -> AliasRecord:
        a = normalize_name(alias)
        t = normalize_name(target)
        if not a or not t:
            raise ValueError("alias and target must be non-empty")
        if a == t:
            rec = self.aliases.get(a) or AliasRecord(a, t, 0, source)
            rec.support += 1
            self.aliases[a] = rec
            return rec
        rec = self.aliases.get(a)
        if rec is None or rec.target != t:
            rec = AliasRecord(a, t, 1, source)
            self.aliases[a] = rec
        else:
            rec.support += 1
        self.observations += 1
        self._prune()
        return rec

    def _prune(self) -> None:
        if len(self.aliases) <= self.max_aliases:
            return
        ranked = sorted(self.aliases.values(), key=lambda r: (r.support, len(r.alias), r.alias))
        for rec in ranked[: len(self.aliases) - self.max_aliases]:
            self.aliases.pop(rec.alias, None)

    def resolve(self, phrase: str, *, max_hops: int = 8) -> Tuple[str, List[str]]:
        cur = normalize_name(phrase)
        trace: List[str] = []
        seen = set()
        for _ in range(max_hops):
            if cur in seen:
                break
            seen.add(cur)
            rec = self.aliases.get(cur)
            if rec is None or rec.target == cur:
                break
            trace.append(f"{cur}->{rec.target}")
            cur = rec.target
        return cur, trace

    def summary(self, limit: int = 20) -> Dict[str, object]:
        rows = sorted(self.aliases.values(), key=lambda r: (-r.support, r.alias))[:limit]
        return {
            "version": self.VERSION,
            "aliases": len(self.aliases),
            "max_aliases": self.max_aliases,
            "observations": self.observations,
            "strongest": [asdict(r) for r in rows],
            "raw_conversations_retained": 0,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "max_aliases": self.max_aliases,
            "observations": self.observations,
            "aliases": {k: asdict(v) for k, v in self.aliases.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "LexicalSemanticMemory":
        obj = cls(int(data.get("max_aliases", 8192)))
        obj.observations = int(data.get("observations", 0))
        for key, row in dict(data.get("aliases", {})).items():
            obj.aliases[str(key)] = AliasRecord(
                alias=str(row.get("alias", key)),
                target=str(row.get("target", "")),
                support=int(row.get("support", 1)),
                source=str(row.get("source", "user")),
            )
        obj._prune()
        return obj

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LexicalSemanticMemory":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class FactRecord:
    subject: str
    relation: str
    object: str
    support: int = 1
    source: str = "user"


class FactMemory:
    """Small bounded semantic fact store for conversational grounding.

    It intentionally supports only explicit inspectable triples in V0.15. This
    is not meant to be a general knowledge graph yet; it is the first conversational
    bridge for facts that do not fit the executable ConceptStore definition model.
    """

    VERSION = "APCN-V0.15-FACT-MEMORY"

    def __init__(self, max_facts: int = 16384):
        self.max_facts = int(max_facts)
        self.facts: Dict[Tuple[str, str, str], FactRecord] = {}
        self.observations = 0

    def teach(self, subject: str, relation: str, obj: str, *, source: str = "user") -> FactRecord:
        s = normalize_name(subject)
        r = normalize_name(relation).replace(" ", "_")
        o = normalize_name(obj)
        if not s or not r or not o:
            raise ValueError("fact components must be non-empty")
        key = (s, r, o)
        rec = self.facts.get(key)
        if rec is None:
            rec = FactRecord(s, r, o, 1, source)
            self.facts[key] = rec
        else:
            rec.support += 1
        self.observations += 1
        self._prune()
        return rec

    def _prune(self) -> None:
        if len(self.facts) <= self.max_facts:
            return
        ranked = sorted(self.facts.values(), key=lambda r: (r.support, r.subject, r.relation, r.object))
        for rec in ranked[: len(self.facts) - self.max_facts]:
            self.facts.pop((rec.subject, rec.relation, rec.object), None)

    def about(self, subject: str) -> List[FactRecord]:
        s = normalize_name(subject)
        rows = [r for r in self.facts.values() if r.subject == s]
        rows.sort(key=lambda r: (-r.support, r.relation, r.object))
        return rows

    def first_is_a(self, subject: str) -> Optional[FactRecord]:
        for rec in self.about(subject):
            if rec.relation == "is_a":
                return rec
        return None

    def summary(self, limit: int = 20) -> Dict[str, object]:
        rows = sorted(self.facts.values(), key=lambda r: (-r.support, r.subject))[:limit]
        return {
            "version": self.VERSION,
            "facts": len(self.facts),
            "max_facts": self.max_facts,
            "observations": self.observations,
            "strongest": [asdict(r) for r in rows],
            "raw_conversations_retained": 0,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "max_facts": self.max_facts,
            "observations": self.observations,
            "facts": [asdict(r) for r in self.facts.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "FactMemory":
        obj = cls(int(data.get("max_facts", 16384)))
        obj.observations = int(data.get("observations", 0))
        for row in list(data.get("facts", [])):
            rec = FactRecord(
                subject=str(row.get("subject", "")),
                relation=str(row.get("relation", "")),
                object=str(row.get("object", "")),
                support=int(row.get("support", 1)),
                source=str(row.get("source", "user")),
            )
            if rec.subject and rec.relation and rec.object:
                obj.facts[(rec.subject, rec.relation, rec.object)] = rec
        obj._prune()
        return obj

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FactMemory":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
