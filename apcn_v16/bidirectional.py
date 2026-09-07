from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import math
import re

from apcn_v10.definitions import ConceptStore, Expr, normalize_name
from apcn_v15.conversation import ConversationReply
from apcn_v15.lexicon import FactMemory, LexicalSemanticMemory


_SURFACE_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)
_SLOT = re.compile(r"\{([a-z_]+)\}")


def _norm_surface(text: str) -> str:
    return " ".join(_SURFACE_TOKEN.findall(str(text).lower())).strip()


def _join_items(items: Sequence[str]) -> str:
    rows = [str(x).strip() for x in items if str(x).strip()]
    if not rows:
        return "nothing recorded"
    if len(rows) == 1:
        return rows[0]
    if len(rows) == 2:
        return f"{rows[0]} and {rows[1]}"
    return ", ".join(rows[:-1]) + f", and {rows[-1]}"


@dataclass(frozen=True)
class SemanticFrame:
    """Small explicit language-independent semantic frame.

    V0.16 intentionally keeps the frame representation inspectable. Language
    constructions may map to/from these frames, but factual truth lives in the
    concept/world memories rather than in the construction memory.
    """

    op: str
    slots: Tuple[Tuple[str, str], ...] = ()
    polarity: bool = True

    @classmethod
    def make(cls, op: str, /, **slots: object) -> "SemanticFrame":
        pairs = tuple(sorted((str(k), normalize_name(str(v)) if k not in {"definition", "dependencies", "contrast", "reason"} else str(v).strip()) for k, v in slots.items()))
        return cls(str(op).upper(), pairs, True)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        for k, v in self.slots:
            if k == key:
                return v
        return default

    def slot_dict(self) -> Dict[str, str]:
        return dict(self.slots)

    def to_dict(self) -> Dict[str, object]:
        return {"op": self.op, "slots": dict(self.slots), "polarity": self.polarity}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SemanticFrame":
        slots = data.get("slots", {})
        if isinstance(slots, Mapping):
            pairs = tuple(sorted((str(k), str(v)) for k, v in slots.items()))
        else:
            pairs = tuple((str(k), str(v)) for k, v in slots)  # type: ignore[arg-type]
        return cls(str(data.get("op", "UNKNOWN")).upper(), pairs, bool(data.get("polarity", True)))


@dataclass
class ConstructionRecord:
    op: str
    template: str
    roles: Tuple[str, ...]
    support: int = 0
    parse_hits: int = 0
    generation_hits: int = 0

    def key(self) -> str:
        return f"{self.op}|{self.template}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "op": self.op,
            "template": self.template,
            "roles": list(self.roles),
            "support": self.support,
            "parse_hits": self.parse_hits,
            "generation_hits": self.generation_hits,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ConstructionRecord":
        return cls(
            str(data["op"]),
            str(data["template"]),
            tuple(str(x) for x in data.get("roles", [])),
            int(data.get("support", 0)),
            int(data.get("parse_hits", 0)),
            int(data.get("generation_hits", 0)),
        )


class BidirectionalConstructionMemory:
    """Bounded learned surface <-> semantic construction memory.

    A paired demonstration supplies a surface utterance and its semantic frame.
    The learner abstracts explicit slot values out of the surface form and keeps
    only the reusable construction. The same retained construction is used in
    both directions: parsing and realization.

    This is intentionally not a language model: it stores no token-prediction
    weights, raw corpus, or world facts. It can only express semantic content
    present in the frame passed to ``generate``.
    """

    VERSION = "APCN-V0.16-BIDIRECTIONAL-CONSTRUCTIONS"

    def __init__(self, max_patterns: int = 2048):
        self.max_patterns = int(max_patterns)
        self.records: Dict[str, ConstructionRecord] = {}
        self.observations = 0
        self.raw_sentences_retained = 0

    @staticmethod
    def _abstract(surface: str, frame: SemanticFrame) -> Tuple[str, Tuple[str, ...]]:
        text = _norm_surface(surface)
        role_values = []
        for role, value in frame.slots:
            value_n = _norm_surface(value)
            if not value_n:
                continue
            role_values.append((len(value_n.split()), len(value_n), role, value_n))
        for _, _, role, value_n in sorted(role_values, reverse=True):
            text = re.sub(rf"(?<![a-z0-9]){re.escape(value_n)}(?![a-z0-9])", "{" + role + "}", text, flags=re.I)
        roles = tuple(m.group(1) for m in _SLOT.finditer(text))
        return text, roles

    def observe(self, surface: str, frame: SemanticFrame, weight: int = 1) -> ConstructionRecord:
        template, roles = self._abstract(surface, frame)
        if not template:
            raise ValueError("empty construction surface")
        # Every semantic slot that is linguistically required in the example must
        # have been abstracted. Frames may contain metadata-like slots not spoken,
        # but a zero-role non-social construction is rejected as memorization.
        if not roles and frame.slots and frame.op not in {"GREETING", "SOCIAL"}:
            raise ValueError(f"surface does not expose any frame slots: {surface!r}")
        key = f"{frame.op}|{template}"
        rec = self.records.get(key)
        if rec is None:
            rec = ConstructionRecord(frame.op, template, tuple(dict.fromkeys(roles)))
            self.records[key] = rec
        rec.support += max(1, int(weight))
        self.observations += max(1, int(weight))
        self._prune()
        return rec

    def _prune(self) -> None:
        if len(self.records) <= self.max_patterns:
            return
        ranked = sorted(self.records.values(), key=lambda r: (r.support + r.parse_hits + r.generation_hits, r.support, r.key()))
        for rec in ranked[: len(self.records) - self.max_patterns]:
            self.records.pop(rec.key(), None)

    @staticmethod
    def _regex_for(template: str) -> re.Pattern[str]:
        parts: List[str] = []
        pos = 0
        seen: set[str] = set()
        for m in _SLOT.finditer(template):
            literal = template[pos:m.start()]
            esc = re.escape(literal).replace(r"\ ", r"\s+")
            parts.append(esc)
            role = m.group(1)
            if role in seen:
                parts.append(rf"(?P={role})")
            else:
                parts.append(rf"(?P<{role}>.+?)")
                seen.add(role)
            pos = m.end()
        literal = template[pos:]
        parts.append(re.escape(literal).replace(r"\ ", r"\s+"))
        return re.compile(r"^" + "".join(parts) + r"$", re.I)

    @staticmethod
    def _literal_weight(template: str) -> float:
        literal = _SLOT.sub(" ", template)
        n = len(_SURFACE_TOKEN.findall(literal))
        return 1.0 + math.log1p(n)

    def parse(self, surface: str, *, allowed_ops: Optional[Iterable[str]] = None) -> Tuple[Optional[SemanticFrame], float, List[Dict[str, object]]]:
        text = _norm_surface(surface)
        allowed = None if allowed_ops is None else {str(x).upper() for x in allowed_ops}
        candidates: List[Tuple[float, ConstructionRecord, SemanticFrame]] = []
        for rec in self.records.values():
            if allowed is not None and rec.op not in allowed:
                continue
            m = self._regex_for(rec.template).match(text)
            if not m:
                continue
            slots = {role: m.group(role).strip() for role in rec.roles if m.groupdict().get(role)}
            frame = SemanticFrame.make(rec.op, **slots)
            score = math.log1p(rec.support) * self._literal_weight(rec.template)
            candidates.append((score, rec, frame))
        if not candidates:
            return None, 0.0, []
        candidates.sort(key=lambda row: (row[0], row[1].support, row[1].template), reverse=True)
        top_score, top_rec, top_frame = candidates[0]
        second = candidates[1][0] if len(candidates) > 1 else 0.0
        margin = top_score / max(top_score + second, 1e-9)
        support_factor = 1.0 - math.exp(-top_rec.support / 3.0)
        confidence = min(.99, .55 * margin + .45 * support_factor)
        top_rec.parse_hits += 1
        evidence = [
            {"op": rec.op, "template": rec.template, "support": rec.support, "score": float(score)}
            for score, rec, _ in candidates[:5]
        ]
        return top_frame, float(confidence), evidence

    @staticmethod
    def _finish(op: str, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return text
        text = text[0].upper() + text[1:]
        if op.startswith("ASK_"):
            return text.rstrip(".?!") + "?"
        return text.rstrip(".?!") + "."

    def generate(self, frame: SemanticFrame, limit: int = 12) -> List[str]:
        slots = frame.slot_dict()
        rows: List[Tuple[int, str, ConstructionRecord]] = []
        for rec in self.records.values():
            if rec.op != frame.op:
                continue
            if any(role not in slots for role in rec.roles):
                continue
            text = rec.template
            ok = True
            for role in rec.roles:
                value = str(slots.get(role, "")).strip()
                if not value:
                    ok = False
                    break
                text = text.replace("{" + role + "}", value)
            if not ok or _SLOT.search(text):
                continue
            rows.append((rec.support, self._finish(frame.op, text), rec))
        rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
        out: List[str] = []
        seen = set()
        for _, text, rec in rows:
            if text.lower() in seen:
                continue
            seen.add(text.lower())
            out.append(text)
            rec.generation_hits += 1
            if len(out) >= max(1, int(limit)):
                break
        return out

    def roundtrip(self, frame: SemanticFrame, limit: int = 12) -> Dict[str, object]:
        generations = self.generate(frame, limit)
        rows = []
        exact = 0
        for text in generations:
            parsed, conf, evidence = self.parse(text, allowed_ops={frame.op})
            ok = parsed == frame
            exact += int(ok)
            rows.append({"surface": text, "parsed": None if parsed is None else parsed.to_dict(), "confidence": conf, "exact": ok, "evidence": evidence[:2]})
        return {
            "semantic": frame.to_dict(),
            "generations": rows,
            "count": len(rows),
            "roundtrip_exact": exact / max(1, len(rows)),
        }

    def summary(self, limit: int = 20) -> Dict[str, object]:
        by_op = Counter(rec.op for rec in self.records.values())
        strongest = sorted(self.records.values(), key=lambda r: (r.support, r.parse_hits + r.generation_hits, r.template), reverse=True)[:limit]
        return {
            "version": self.VERSION,
            "observations": self.observations,
            "patterns": len(self.records),
            "max_patterns": self.max_patterns,
            "by_op": dict(by_op),
            "strongest": [r.to_dict() for r in strongest],
            "raw_sentences_retained": 0,
            "world_facts_retained": 0,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "max_patterns": self.max_patterns,
            "observations": self.observations,
            "records": [r.to_dict() for r in self.records.values()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "BidirectionalConstructionMemory":
        obj = cls(int(data.get("max_patterns", 2048)))
        obj.observations = int(data.get("observations", 0))
        for row in data.get("records", []):
            rec = ConstructionRecord.from_dict(row)  # type: ignore[arg-type]
            obj.records[rec.key()] = rec
        obj._prune()
        return obj

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BidirectionalConstructionMemory":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class BidirectionalTeacherV16:
    """Paired semantic/surface demonstrations for V0.16.

    TRAIN is the only material used to bootstrap the construction memory. DEV is
    frozen diagnostic material and is never observed by ``bootstrap``.
    """

    TRAIN: Dict[str, Tuple[str, ...]] = {
        "ASK_DEFINITION": (
            "what is {concept}",
            "define {concept}",
            "explain {concept}",
            "give me the meaning of {concept}",
            "tell me what {concept} means",
        ),
        "ASK_DEPENDENCIES": (
            "what does {concept} depend on",
            "which concepts support {concept}",
            "list the dependencies of {concept}",
            "what is {concept} built from",
            "which ideas underlie {concept}",
        ),
        "ASK_KNOWLEDGE": (
            "do you know {concept}",
            "do you understand {concept}",
            "are you familiar with {concept}",
            "is {concept} in your explicit memory",
            "have you learned {concept}",
        ),
        "COMPARE": (
            "compare {left} and {right}",
            "contrast {left} with {right}",
            "how do {left} and {right} differ",
            "tell me the difference between {left} and {right}",
            "compare the concepts {left} and {right}",
        ),
        "STATE_DEFINITION": (
            "{concept} is {definition}",
            "{concept} can be defined as {definition}",
            "in my explicit memory {concept} means {definition}",
            "put simply {concept} is {definition}",
            "the stored definition of {concept} is {definition}",
            "i understand {concept} as {definition}",
        ),
        "STATE_DEPENDENCIES": (
            "{concept} depends on {dependencies}",
            "the direct dependencies of {concept} are {dependencies}",
            "{dependencies} are the concepts directly supporting {concept}",
            "in my concept memory {concept} uses {dependencies}",
            "for {concept} i have the direct inputs {dependencies}",
            "the stored structure for {concept} points to {dependencies}",
        ),
        "STATE_KNOWN": (
            "yes i have explicit knowledge of {concept}",
            "yes {concept} exists in my explicit concept memory",
            "i currently know {concept}",
            "{concept} is represented in my stored knowledge",
            "i have an explicit memory entry for {concept}",
        ),
        "STATE_UNKNOWN": (
            "i do not currently know {concept}",
            "{concept} is not in my explicit knowledge yet",
            "i do not have a stored concept for {concept}",
            "my current memory has no concept for {concept}",
            "i need teaching or grounding before i can claim {concept}",
        ),
        "STATE_COMPARE": (
            "{left} and {right} differ in this way {contrast}",
            "comparing {left} with {right} {contrast}",
            "the explicit structural contrast between {left} and {right} is {contrast}",
            "for {left} versus {right} my stored comparison is {contrast}",
            "in my concept graph {left} and {right} compare as follows {contrast}",
        ),
        "CLARIFY_UNKNOWN": (
            "i cannot answer about {concept} because it is not in my explicit memory",
            "i need you to teach or ground {concept} before i answer",
            "i do not have enough stored knowledge about {concept} yet",
            "{concept} is currently unknown to my concept memory",
            "i cannot safely invent an answer for {concept}",
        ),
    }

    DEV: Dict[str, Tuple[str, ...]] = {
        "ASK_DEFINITION": (
            "how would you describe {concept}",
            "what does {concept} mean in your knowledge",
        ),
        "ASK_DEPENDENCIES": (
            "what conceptual pieces feed {concept}",
            "name what {concept} relies upon",
        ),
        "ASK_KNOWLEDGE": (
            "can you claim explicit knowledge of {concept}",
            "is {concept} something you currently understand",
        ),
        "COMPARE": (
            "set {left} against {right} conceptually",
            "what separates {left} from {right}",
        ),
    }

    CONCEPTS = ("acceleration", "speed", "density", "force", "pressure", "momentum")

    @staticmethod
    def frame_for(op: str, *, concept: str = "acceleration", left: str = "speed", right: str = "density") -> SemanticFrame:
        if op in {"ASK_DEFINITION", "ASK_DEPENDENCIES", "ASK_KNOWLEDGE"}:
            return SemanticFrame.make(op, concept=concept)
        if op == "COMPARE":
            return SemanticFrame.make(op, left=left, right=right)
        if op == "STATE_DEFINITION":
            return SemanticFrame.make(op, concept=concept, definition="velocity change divided by time")
        if op == "STATE_DEPENDENCIES":
            return SemanticFrame.make(op, concept=concept, dependencies="velocity change and time")
        if op in {"STATE_KNOWN", "STATE_UNKNOWN", "CLARIFY_UNKNOWN"}:
            return SemanticFrame.make(op, concept=concept)
        if op == "STATE_COMPARE":
            return SemanticFrame.make(op, left=left, right=right, contrast="they have different direct dependencies")
        raise KeyError(op)

    def bootstrap(self, memory: BidirectionalConstructionMemory, repeats: int = 4) -> Dict[str, object]:
        before = memory.observations
        for op, templates in self.TRAIN.items():
            frame = self.frame_for(op)
            for template in templates:
                surface = template.format(**frame.slot_dict())
                memory.observe(surface, frame, weight=max(1, int(repeats)))
        return {
            "observations_added": memory.observations - before,
            "patterns": len(memory.records),
            "train_templates": sum(len(v) for v in self.TRAIN.values()),
            "dev_templates_used": 0,
        }


class SemanticResponderV16:
    """Reason over explicit APCN knowledge and return a semantic answer frame.

    The realizer never receives ``ConceptStore`` or ``FactMemory``. This class is
    the firewall between factual reasoning and surface generation.
    """

    def __init__(self, concepts: ConceptStore, lexicon: LexicalSemanticMemory, facts: FactMemory):
        self.concepts = concepts
        self.lexicon = lexicon
        self.facts = facts

    @staticmethod
    def _expr_surface(expr: Expr) -> str:
        def render(node: Expr) -> str:
            if node.op == "REF":
                return str(node.args[0])
            args = [render(a) if isinstance(a, Expr) else str(a) for a in node.args]
            if node.op == "MUL" and len(args) == 2:
                return f"{args[0]} multiplied by {args[1]}"
            if node.op == "DIV" and len(args) == 2:
                return f"{args[0]} divided by {args[1]}"
            if node.op == "ADD" and len(args) == 2:
                return f"{args[0]} plus {args[1]}"
            if node.op == "SUB" and len(args) == 2:
                return f"{args[0]} minus {args[1]}"
            if node.op == "RATE" and len(args) == 2:
                return f"the rate of change of {args[0]} with respect to {args[1]}"
            if node.op == "AND":
                return _join_items(args)
            if node.op == "IS_A" and len(args) == 1:
                return f"a kind of {args[0]}"
            return node.pretty()
        return render(expr)

    def _resolve(self, name: str) -> str:
        resolved, _ = self.lexicon.resolve(name)
        return normalize_name(resolved)

    def answer(self, request: SemanticFrame) -> SemanticFrame:
        op = request.op
        if op == "ASK_DEFINITION":
            concept = self._resolve(request.get("concept", "") or "")
            rec = self.concepts.records.get(concept)
            if rec is not None:
                if rec.definition is not None:
                    definition = self._expr_surface(rec.definition)
                elif rec.primitive:
                    definition = "a grounded primitive concept" if rec.grounded else "a primitive concept whose grounding is incomplete"
                else:
                    definition = "an explicitly stored concept"
                return SemanticFrame.make("STATE_DEFINITION", concept=concept, definition=definition)
            facts = self.facts.about(concept)
            if facts:
                first = facts[0]
                if first.relation == "is_a":
                    return SemanticFrame.make("STATE_DEFINITION", concept=concept, definition=f"a {first.object}")
            return SemanticFrame.make("CLARIFY_UNKNOWN", concept=concept)

        if op == "ASK_DEPENDENCIES":
            concept = self._resolve(request.get("concept", "") or "")
            rec = self.concepts.records.get(concept)
            if rec is None:
                return SemanticFrame.make("CLARIFY_UNKNOWN", concept=concept)
            deps = sorted(rec.dependencies())
            return SemanticFrame.make("STATE_DEPENDENCIES", concept=concept, dependencies=_join_items(deps))

        if op == "ASK_KNOWLEDGE":
            concept = self._resolve(request.get("concept", "") or "")
            known = concept in self.concepts.records or bool(self.facts.about(concept))
            return SemanticFrame.make("STATE_KNOWN" if known else "STATE_UNKNOWN", concept=concept)

        if op == "COMPARE":
            left = self._resolve(request.get("left", "") or "")
            right = self._resolve(request.get("right", "") or "")
            lrec = self.concepts.records.get(left)
            rrec = self.concepts.records.get(right)
            missing = [name for name, rec in ((left, lrec), (right, rrec)) if rec is None]
            if missing:
                return SemanticFrame.make("CLARIFY_UNKNOWN", concept=_join_items(missing))
            ldeps = set(lrec.dependencies()) if lrec else set()
            rdeps = set(rrec.dependencies()) if rrec else set()
            shared = sorted(ldeps & rdeps)
            only_l = sorted(ldeps - rdeps)
            only_r = sorted(rdeps - ldeps)
            pieces: List[str] = []
            if shared:
                pieces.append(f"both use {_join_items(shared)}")
            if only_l:
                pieces.append(f"only {left} directly uses {_join_items(only_l)}")
            if only_r:
                pieces.append(f"only {right} directly uses {_join_items(only_r)}")
            if not pieces:
                pieces.append("their current direct dependency sets do not distinguish them")
            return SemanticFrame.make("STATE_COMPARE", left=left, right=right, contrast="; ".join(pieces))

        return SemanticFrame.make("CLARIFY_UNKNOWN", concept=request.get("concept", "that request") or "that request")


class BidirectionalLanguageEngineV16:
    """V0.16 semantic conversation path with V0.15 fallback for unsupported acts."""

    VERSION = "APCN-V0.16-BIDIRECTIONAL-LANGUAGE-ENGINE"
    REQUEST_OPS = {"ASK_DEFINITION", "ASK_DEPENDENCIES", "ASK_KNOWLEDGE", "COMPARE"}
    ACT_MAP = {
        "STATE_DEFINITION": "ANSWER_DEFINITION",
        "STATE_DEPENDENCIES": "ANSWER_DEPENDENCIES",
        "STATE_KNOWN": "ANSWER_KNOWLEDGE",
        "STATE_UNKNOWN": "ANSWER_KNOWLEDGE",
        "STATE_COMPARE": "ANSWER_COMPARE",
        "CLARIFY_UNKNOWN": "CLARIFY",
    }

    def __init__(self, memory: BidirectionalConstructionMemory, responder: SemanticResponderV16, *, fallback_engine=None):
        self.memory = memory
        self.responder = responder
        self.fallback_engine = fallback_engine
        self.turn = 0
        self.last_request: Optional[SemanticFrame] = None
        self.last_answer: Optional[SemanticFrame] = None
        self.last_parse_evidence: List[Dict[str, object]] = []

    def parse(self, text: str) -> Tuple[Optional[SemanticFrame], float, List[Dict[str, object]]]:
        frame, confidence, evidence = self.memory.parse(text, allowed_ops=self.REQUEST_OPS)
        self.last_parse_evidence = evidence
        return frame, confidence, evidence

    def paraphrases(self, text: str, limit: int = 10) -> Dict[str, object]:
        frame, confidence, evidence = self.parse(text)
        if frame is None:
            return {"parsed": False, "confidence": confidence, "semantic": None, "paraphrases": [], "evidence": evidence}
        return {
            "parsed": True,
            "confidence": confidence,
            "semantic": frame.to_dict(),
            "paraphrases": self.memory.generate(frame, limit),
            "evidence": evidence,
        }

    def respond(self, text: str) -> ConversationReply:
        frame, confidence, evidence = self.parse(text)
        if frame is None or confidence < .52:
            if self.fallback_engine is not None:
                row = self.fallback_engine.respond(text)
                row.trace.append("v016:fallback_to_v015")
                return row
            return ConversationReply("I cannot yet map that wording into a reliable semantic operation.", "CLARIFY", .25, trace=["v016:no_parse"])

        answer = self.responder.answer(frame)
        surfaces = self.memory.generate(answer, limit=12)
        if not surfaces:
            if self.fallback_engine is not None:
                row = self.fallback_engine.respond(text)
                row.trace.append(f"v016:no_realizer:{answer.op}")
                return row
            return ConversationReply("I formed a semantic answer but do not yet know how to express it safely.", "CLARIFY", .30, trace=[f"v016:no_realizer:{answer.op}"])

        chosen = surfaces[self.turn % len(surfaces)]
        self.turn += 1
        self.last_request = frame
        self.last_answer = answer
        concept = answer.get("concept") or frame.get("concept") or frame.get("left")
        trace = [
            f"v016:parse:{frame.op}:{confidence:.3f}",
            f"v016:reason:{answer.op}",
            "v016:realize:construction_memory_only",
        ]
        for row in evidence[:2]:
            trace.append(f"v016:evidence:{row['template']}")
        return ConversationReply(chosen, self.ACT_MAP.get(answer.op, answer.op), confidence, concept=concept, trace=trace)

    def summary(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "turn": self.turn,
            "last_request": None if self.last_request is None else self.last_request.to_dict(),
            "last_answer": None if self.last_answer is None else self.last_answer.to_dict(),
            "construction_memory": self.memory.summary(12),
            "generator_has_world_memory_reference": False,
        }
