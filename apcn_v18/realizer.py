from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple
import json
import re


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


@dataclass(frozen=True)
class AnswerPlanV18:
    """Truth-authorized semantic answer plan."""

    act: str
    slots: Tuple[Tuple[str, str], ...] = ()
    confidence: float = 1.0

    @classmethod
    def make(cls, act: str, /, *, confidence: float = 1.0, **slots: object) -> "AnswerPlanV18":
        pairs = tuple(sorted((str(k), _norm(str(v))) for k, v in slots.items() if str(v).strip()))
        return cls(str(act).upper(), pairs, float(confidence))

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        for k, v in self.slots:
            if k == key:
                return v
        return default

    def slot_dict(self) -> Dict[str, str]:
        return dict(self.slots)

    def to_dict(self) -> Dict[str, object]:
        return {"act": self.act, "slots": dict(self.slots), "confidence": self.confidence}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "AnswerPlanV18":
        slots = data.get("slots", {})
        if isinstance(slots, Mapping):
            pairs = tuple(sorted((str(k), str(v)) for k, v in slots.items()))
        else:
            pairs = tuple((str(k), str(v)) for k, v in slots)  # type: ignore[arg-type]
        return cls(str(data.get("act", "CLARIFY")).upper(), pairs, float(data.get("confidence", 1.0)))


@dataclass
class RealizationRecordV18:
    act: str
    template: str
    support: int = 1
    uses: int = 0

    def key(self) -> str:
        return f"{self.act}|{self.template}"

    def to_dict(self) -> Dict[str, object]:
        return {"act": self.act, "template": self.template, "support": self.support, "uses": self.uses}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RealizationRecordV18":
        return cls(str(data["act"]).upper(), str(data["template"]), int(data.get("support", 1)), int(data.get("uses", 0)))


class NaturalRealizationMemoryV18:
    """Bounded semantic-plan -> English construction memory.

    It stores no facts, world state, concept graph, raw dialogue, token-prediction
    weights, or external model. It may only realize content already present in an
    AnswerPlanV18.
    """

    VERSION = "APCN-V0.18-NATURAL-REALIZATION-MEMORY"

    BOOTSTRAP: Dict[str, Tuple[str, ...]] = {
        "ANSWER_TRUE_EXPLICIT": (
            "Yes. {statement}.",
            "Yes — {statement}.",
            "Yes. My explicit memory supports that {statement}.",
        ),
        "ANSWER_TRUE_DERIVED": (
            "Yes. {statement}. I can derive that because {reason}.",
            "Yes — {statement}. The reason is that {reason}.",
            "Yes. {reason}, so {statement}.",
        ),
        "ANSWER_FALSE": (
            "No. My explicit memory supports {negation}.",
            "No. I have evidence for the opposite: {negation}.",
        ),
        "ANSWER_UNKNOWN": (
            "I don't know whether {statement} yet. I have no explicit fact or rule that establishes it.",
            "I can't confirm that {statement}. My current memory has no supporting or negating proof.",
        ),
        "ANSWER_ABOUT": (
            "About {subject}, I currently know: {facts}.",
            "I have {count} relevant statements about {subject}: {facts}.",
        ),
        "ANSWER_CAUSE": (
            "According to my explicit memory, {effect} because {cause}.",
            "The cause I have stored for {effect} is {cause}.",
        ),
        "ANSWER_CONSEQUENCE": (
            "If {condition}, then {consequence}.",
            "Given {condition}, my stored rule says {consequence}.",
        ),
        "ANSWER_BEFORE": (
            "{first} happens before {second}.",
            "My temporal memory places {first} before {second}.",
        ),
        "ANSWER_WHY": (
            "Because {reason}.",
            "The reason is that {reason}.",
        ),
        "ANSWER_DEFINITION": (
            "{concept} is {definition}.",
            "Put simply, {concept} is {definition}.",
            "My stored definition of {concept} is {definition}.",
        ),
        "CLARIFY_SUGGEST": (
            "I don't know '{term}'. Did you mean '{suggestion}'?",
            "I don't have a meaning for '{term}'. I do know '{suggestion}'; did you mean that?",
        ),
        "CLARIFY_UNKNOWN_TERM": (
            "I don't know '{term}' yet. You can teach it as an alias, definition, or fact.",
            "'{term}' is not in my explicit language or concept memory yet.",
        ),
        "ACK_SEMANTIC_TEACH": (
            "Stored as explicit semantic memory: {statement}.",
            "Understood. I stored the semantic statement: {statement}.",
        ),
    }

    def __init__(self, max_patterns: int = 512, *, bootstrap: bool = True):
        self.max_patterns = int(max_patterns)
        self.records: Dict[str, RealizationRecordV18] = {}
        self.observations = 0
        self.raw_sentences_retained = 0
        if bootstrap:
            for act, templates in self.BOOTSTRAP.items():
                for template in templates:
                    self.observe(act, template, weight=4)

    @staticmethod
    def _roles(template: str) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(re.findall(r"\{([a-z_]+)\}", template)))

    def observe(self, act: str, template: str, *, weight: int = 1) -> RealizationRecordV18:
        act_n = str(act).upper()
        template_n = _norm(template)
        if not template_n:
            raise ValueError("empty realization template")
        rec = RealizationRecordV18(act_n, template_n, max(1, int(weight)))
        old = self.records.get(rec.key())
        if old is None:
            self.records[rec.key()] = rec
            old = rec
        else:
            old.support += max(1, int(weight))
        self.observations += max(1, int(weight))
        self._prune()
        return old

    def _prune(self) -> None:
        if len(self.records) <= self.max_patterns:
            return
        ranked = sorted(self.records.values(), key=lambda r: (r.support + r.uses, r.support, r.key()))
        for rec in ranked[: len(self.records) - self.max_patterns]:
            self.records.pop(rec.key(), None)

    @staticmethod
    def _finish(text: str) -> str:
        text = _norm(text)
        text = re.sub(r"\s+([,.!?])", r"\1", text)
        text = re.sub(r"\.{2,}", ".", text)
        if not text:
            return text
        text = text[0].upper() + text[1:]
        if text[-1] not in ".?!":
            text += "."
        return text

    def generate(self, plan: AnswerPlanV18, limit: int = 8) -> List[str]:
        slots = plan.slot_dict()
        rows: List[Tuple[int, str, RealizationRecordV18]] = []
        for rec in self.records.values():
            if rec.act != plan.act:
                continue
            roles = self._roles(rec.template)
            if any(not slots.get(role) for role in roles):
                continue
            text = rec.template
            for role in roles:
                text = text.replace("{" + role + "}", slots[role].rstrip(" .?!"))
            if re.search(r"\{[a-z_]+\}", text):
                continue
            rows.append((rec.support, self._finish(text), rec))
        rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
        out: List[str] = []
        seen = set()
        for _, text, rec in rows:
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
            rec.uses += 1
            if len(out) >= max(1, int(limit)):
                break
        return out

    def summary(self, limit: int = 20) -> Dict[str, object]:
        counts: Dict[str, int] = {}
        for rec in self.records.values():
            counts[rec.act] = counts.get(rec.act, 0) + 1
        strongest = sorted(self.records.values(), key=lambda r: (r.support, r.uses, r.template), reverse=True)[:limit]
        return {
            "version": self.VERSION,
            "patterns": len(self.records),
            "max_patterns": self.max_patterns,
            "observations": self.observations,
            "by_act": counts,
            "strongest": [r.to_dict() for r in strongest],
            "raw_sentences_retained": 0,
            "world_facts_retained": 0,
            "truth_memory_reference": False,
            "external_llm": False,
            "neural_language_model": False,
        }

    def to_dict(self) -> Dict[str, object]:
        return {"version": self.VERSION, "max_patterns": self.max_patterns, "observations": self.observations, "records": [r.to_dict() for r in self.records.values()]}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "NaturalRealizationMemoryV18":
        obj = cls(int(data.get("max_patterns", 512)), bootstrap=False)
        obj.observations = int(data.get("observations", 0))
        for row in data.get("records", []):
            rec = RealizationRecordV18.from_dict(row)  # type: ignore[arg-type]
            obj.records[rec.key()] = rec
        obj._prune()
        return obj

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NaturalRealizationMemoryV18":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class NaturalRealizerV18:
    """Surface renderer with no access to APCN truth/world/concept memories."""

    VERSION = "APCN-V0.18-NATURAL-REALIZER"

    def __init__(self, memory: NaturalRealizationMemoryV18):
        self.memory = memory
        self.turn = 0
        self.last_plan: Optional[AnswerPlanV18] = None

    def realize(self, plan: AnswerPlanV18) -> str:
        rows = self.memory.generate(plan, 12)
        self.last_plan = plan
        if not rows:
            return "I formed an answer plan but do not yet have a safe English realization for it."
        chosen = rows[self.turn % len(rows)]
        self.turn += 1
        return chosen

    def variants(self, plan: AnswerPlanV18, limit: int = 12) -> List[str]:
        return self.memory.generate(plan, limit)

    def summary(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "turn": self.turn,
            "last_plan": None if self.last_plan is None else self.last_plan.to_dict(),
            "memory": self.memory.summary(16),
            "has_truth_memory_reference": False,
            "has_concept_memory_reference": False,
            "has_world_memory_reference": False,
        }
