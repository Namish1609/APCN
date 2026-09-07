from __future__ import annotations

from collections import Counter, defaultdict
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple
import math
import re

from apcn_v10.definitions import normalize_name
from .bidirectional import (
    SemanticFrame,
    BidirectionalConstructionMemory,
    BidirectionalLanguageEngineV16,
    SemanticResponderV16,
    _SLOT,
    _SURFACE_TOKEN,
    _norm_surface,
)


# Frozen before first execution. Every content cue in this split occurs in the
# original TRAIN constructions, but the ordering/composition is new. This tests
# recombination of learned linguistic functions rather than zero-shot knowledge
# of entirely unseen vocabulary.
RECOMBINATION_SPLIT: Dict[str, Tuple[str, ...]] = {
    "ASK_DEFINITION": (
        "explain what {concept} means",
        "tell me the meaning of {concept}",
        "define and explain {concept}",
    ),
    "ASK_DEPENDENCIES": (
        "tell me which concepts {concept} depends on",
        "explain the dependencies of {concept}",
        "which ideas does {concept} depend on",
    ),
    "ASK_KNOWLEDGE": (
        "do you understand and know {concept}",
        "is {concept} familiar to you",
        "have you learned and understood {concept}",
    ),
    "COMPARE": (
        "tell me how {left} and {right} differ",
        "explain the difference between {left} and {right}",
        "compare and contrast {left} with {right}",
    ),
}


class CompositionalRequestParserV16:
    """Recombine semantic cues learned from TRAIN constructions.

    Exact bidirectional constructions remain the strongest path. This parser is
    used only when no trustworthy exact learned construction matches. It derives
    cue->operation evidence from the existing construction memory and resolves
    semantic slots from explicit known concept/alias names. There is no hand-
    written word->intent dictionary.

    A deliberately small morphology normalizer collapses common English surface
    inflections before cue comparison. It does not assign semantics: the same
    normalization is applied to both learned TRAIN cues and new input tokens.
    """

    STOP = {
        "a","an","the","is","are","was","were","be","been","being",
        "i","me","my","you","your","we","our","it","its","that","this",
        "what","which","who","how","do","does","did","can","could","would","should",
        "and","or","of","to","for","on","in","with","from","at","by","as",
        "please","then","now","just","some","any","there","here","between",
    }

    REQUEST_OPS = {"ASK_DEFINITION", "ASK_DEPENDENCIES", "ASK_KNOWLEDGE", "COMPARE"}

    def __init__(self, memory: BidirectionalConstructionMemory):
        self.memory = memory

    @staticmethod
    def _morph(token: str) -> str:
        """Conservative inflection normalization with no semantic mapping.

        This is intentionally much smaller than a dictionary lemmatizer. It only
        removes productive suffixes needed to compare a learned lexical cue with
        ordinary inflected use: concepts/concept, depends/depend,
        dependencies/dependency, learned/learn, etc. Irregular forms remain
        unknown unless independently learned.
        """
        t = str(token).lower().strip()
        if len(t) > 4 and t.endswith("ies"):
            return t[:-3] + "y"
        if len(t) > 4 and t.endswith("ed"):
            base = t[:-2]
            # studied -> study; learned -> learn. Do not invent irregular roots.
            if base.endswith("i"):
                base = base[:-1] + "y"
            return base
        if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
            return t[:-1]
        return t

    @classmethod
    def _literal_tokens(cls, template: str) -> List[str]:
        literal = _SLOT.sub(" ", template)
        return [cls._morph(t) for t in _SURFACE_TOKEN.findall(literal)]

    def _cue_table(self):
        cue_op: DefaultDict[str, Counter] = defaultdict(Counter)
        op_totals: Counter = Counter()
        for rec in self.memory.records.values():
            if rec.op not in self.REQUEST_OPS:
                continue
            op_totals[rec.op] += rec.support
            for token in set(self._literal_tokens(rec.template)):
                if token in self.STOP or len(token) <= 2:
                    continue
                cue_op[token][rec.op] += rec.support
        return cue_op, op_totals

    @staticmethod
    def _mentioned_terms(text: str, known_terms: Sequence[str]) -> List[str]:
        norm = " " + _norm_surface(text) + " "
        found: List[Tuple[int, int, str]] = []
        occupied: List[Tuple[int, int]] = []
        for term in sorted({normalize_name(x) for x in known_terms if normalize_name(x)}, key=len, reverse=True):
            m = re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", norm, flags=re.I)
            if not m:
                continue
            span = (m.start(), m.end())
            if any(not (span[1] <= a or span[0] >= b) for a, b in occupied):
                continue
            occupied.append(span)
            found.append((span[0], span[1], term))
        found.sort()
        return [row[2] for row in found]

    def parse(self, text: str, known_terms: Sequence[str]) -> Tuple[Optional[SemanticFrame], float, List[Dict[str, object]]]:
        cues, op_totals = self._cue_table()
        total_all = max(1, sum(op_totals.values()))
        scores: Dict[str, float] = defaultdict(float)
        evidence: DefaultDict[str, List[Tuple[float, str]]] = defaultdict(list)
        tokens = [
            self._morph(t)
            for t in _SURFACE_TOKEN.findall(_norm_surface(text))
            if t not in self.STOP
        ]
        for token in tokens:
            if token in self.STOP:
                continue
            row = cues.get(token)
            if not row:
                continue
            support = sum(row.values())
            if support <= 0:
                continue
            for op, count in row.items():
                purity = count / support
                baseline = op_totals.get(op, 0) / total_all
                discrimination = purity - baseline
                if discrimination <= .08:
                    continue
                weight = discrimination * math.log1p(count)
                scores[op] += weight
                evidence[op].append((weight, token))
        if not scores:
            return None, 0.0, []
        ordered = sorted(((score, op) for op, score in scores.items()), reverse=True)
        top, op = ordered[0]
        second = ordered[1][0] if len(ordered) > 1 else 0.0
        terms = self._mentioned_terms(text, known_terms)
        if op == "COMPARE":
            if len(terms) < 2:
                return None, 0.0, []
            frame = SemanticFrame.make(op, left=terms[0], right=terms[1])
        else:
            if not terms:
                return None, 0.0, []
            frame = SemanticFrame.make(op, concept=terms[0])
        ratio = top / max(top + second, 1e-9)
        cue_count = len(evidence[op])
        confidence = min(.91, .48 + .32 * ratio + .08 * min(1.0, cue_count / 2.0))
        ev = [
            {
                "mode":"compositional_cue",
                "op":op,
                "cue":cue,
                "template":f"cue:{cue}",
                "weight":float(weight),
            }
            for weight, cue in sorted(evidence[op], reverse=True)[:6]
        ]
        return frame, float(confidence), ev


class BidirectionalLanguageEngineV161(BidirectionalLanguageEngineV16):
    """V0.16 engine with exact construction + learned cue recombination."""

    VERSION = "APCN-V0.16.1-COMPOSITIONAL-BIDIRECTIONAL-ENGINE"

    def __init__(self, memory: BidirectionalConstructionMemory, responder: SemanticResponderV16, *, fallback_engine=None):
        super().__init__(memory, responder, fallback_engine=fallback_engine)
        self.compositional_parser = CompositionalRequestParserV16(memory)

    def _known_terms(self) -> List[str]:
        terms = set(self.responder.concepts.records)
        terms.update(self.responder.lexicon.aliases)
        terms.update(rec.target for rec in self.responder.lexicon.aliases.values())
        for rec in self.responder.facts.facts.values():
            terms.add(rec.subject); terms.add(rec.object)
        return sorted((normalize_name(x) for x in terms if normalize_name(x)), key=len, reverse=True)

    def _exact_slots_are_resolved(self, frame: SemanticFrame) -> bool:
        known = set(self._known_terms())
        if frame.op == "COMPARE":
            return (frame.get("left") or "") in known and (frame.get("right") or "") in known
        concept = frame.get("concept") or ""
        return concept in known

    def _is_explicit_teaching(self, text: str) -> bool:
        engine = self.fallback_engine
        if engine is None:
            return False
        q = engine._clean(text)
        if any(pattern.match(q) for pattern in engine.ALIAS_PATTERNS):
            return True
        if engine.EXPLICIT_DEFINITION.match(q) or engine.REMEMBER_ISA.match(q):
            return True
        if engine._EXEC_DEFINITION_CUES.search(" " + q + " ") and not q.endswith("?"):
            return True
        return False

    def parse(self, text: str):
        frame, confidence, evidence = self.memory.parse(text, allowed_ops=self.REQUEST_OPS)
        # A slot wildcard is not allowed to swallow arbitrary surrounding syntax.
        # Exact patterns only commit when their captured semantic slots resolve to
        # explicit known terms; otherwise compositional analysis gets a chance.
        if frame is not None and self._exact_slots_are_resolved(frame):
            self.last_parse_evidence = evidence
            return frame, confidence, evidence
        frame, confidence, evidence = self.compositional_parser.parse(text, self._known_terms())
        self.last_parse_evidence = evidence
        return frame, confidence, evidence

    def respond(self, text: str):
        # Teaching changes semantic memory and therefore has priority over request
        # interpretation. This preserves V0.15's immediate explicit teaching path.
        if self._is_explicit_teaching(text) and self.fallback_engine is not None:
            row = self.fallback_engine.respond(text)
            row.trace.append("v016:explicit_semantic_teaching")
            return row
        return super().respond(text)
