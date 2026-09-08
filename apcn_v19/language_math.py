from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import re


OPERATIONS = ("ADD", "SUB", "RSUB")


def _fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return f"{value:.12g}"


def _apply(op: str, a: float, b: float) -> float:
    if op == "ADD":
        return a + b
    if op == "SUB":
        return a - b
    if op == "RSUB":
        return b - a
    raise ValueError(f"unknown operation {op!r}")


@dataclass
class ArithmeticInterpretationV19:
    understood: bool
    text: str
    left: Optional[float] = None
    right: Optional[float] = None
    operation: Optional[str] = None
    result: Optional[float] = None
    confidence: float = 0.0
    evidence: Tuple[str, ...] = ()
    scores: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "understood": self.understood,
            "text": self.text,
            "left": self.left,
            "right": self.right,
            "operation": self.operation,
            "result": self.result,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "scores": dict(self.scores or {}),
        }


class OnlineConstructionArithmeticV19:
    """Online construction learner for two-argument arithmetic language.

    No gradient training is used. Demonstrations constrain a small operation
    hypothesis set (ADD, SUB, reverse-order SUB). Surface n-gram evidence is
    accumulated online. At inference time, learned lexical/construction evidence
    chooses the operation; Python arithmetic executes the selected semantic op.

    The arithmetic words themselves are not hard-coded in the parser. They enter
    memory through demonstrations in `bootstrap_english()` or later user teaching.
    """

    VERSION = "APCN-V0.19-ONLINE-CONSTRUCTION-ARITHMETIC"

    _CONTRACTIONS = {
        "what's": "what is",
        "whats": "what is",
        "how's": "how is",
        "cant": "cannot",
        "can't": "cannot",
        "isn't": "is not",
        "doesn't": "does not",
    }

    _SMALL_NUMBERS = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }

    def __init__(self, *, max_features: int = 8192, bootstrap: bool = True):
        self.max_features = int(max_features)
        self.feature_counts: Dict[str, Dict[str, int]] = {}
        self.demonstrations: List[Dict[str, object]] = []
        self.unresolved_demonstrations = 0
        self.learned_turns = 0
        self.inference_turns = 0
        self.bootstrap_count = 0
        if bootstrap:
            self.bootstrap_english()

    @classmethod
    def normalize(cls, text: str) -> str:
        q = str(text).strip().lower()
        for old, new in cls._CONTRACTIONS.items():
            q = q.replace(old, new)
        q = re.sub(r"[,;:!?()\[\]{}]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    @classmethod
    def tokens(cls, text: str) -> List[str]:
        q = cls.normalize(text)
        return re.findall(r"\d+(?:\.\d+)?|[a-z]+|[+\-=]", q)

    @classmethod
    def _token_value(cls, token: str) -> Optional[float]:
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            return float(token)
        if token in cls._SMALL_NUMBERS:
            return float(cls._SMALL_NUMBERS[token])
        return None

    @classmethod
    def extract_numbers(cls, text: str) -> Tuple[List[float], List[str]]:
        toks = cls.tokens(text)
        values: List[float] = []
        semantic_tokens: List[str] = []
        for token in toks:
            value = cls._token_value(token)
            if value is None:
                semantic_tokens.append(token)
            else:
                values.append(value)
                semantic_tokens.append("<num>")
        return values, semantic_tokens

    @staticmethod
    def _features(tokens: Sequence[str]) -> List[str]:
        lexical = [t for t in tokens if t != "<num>" and t not in {"=", "-"}]
        feats = set()
        for n in (1, 2, 3):
            for i in range(0, len(lexical) - n + 1):
                feats.add(" ".join(lexical[i : i + n]))
        # Symbols are meaningful construction evidence and cannot be dropped.
        for token in tokens:
            if token in {"+", "-"}:
                feats.add(f"symbol:{token}")
        return sorted(feats)

    @staticmethod
    def _candidate_ops(a: float, b: float, result: float) -> List[str]:
        out = []
        for op in OPERATIONS:
            if math.isclose(_apply(op, a, b), result, rel_tol=1e-12, abs_tol=1e-12):
                out.append(op)
        return out

    def _trim_features(self) -> None:
        if len(self.feature_counts) <= self.max_features:
            return
        ranked = sorted(
            self.feature_counts,
            key=lambda f: (sum(self.feature_counts[f].values()), len(f)),
            reverse=True,
        )
        keep = set(ranked[: self.max_features])
        self.feature_counts = {k: v for k, v in self.feature_counts.items() if k in keep}

    def observe(self, surface: str, result: float, *, source: str = "user", weight: int = 1) -> Dict[str, object]:
        numbers, tokenized = self.extract_numbers(surface)
        if len(numbers) != 2:
            return {
                "learned": False,
                "reason": f"expected exactly two input numbers, found {len(numbers)}",
                "surface": self.normalize(surface),
            }
        a, b = numbers
        result = float(result)
        candidates = self._candidate_ops(a, b, result)
        row = {
            "surface": self.normalize(surface),
            "left": a,
            "right": b,
            "result": result,
            "candidate_operations": list(candidates),
            "source": source,
        }
        self.demonstrations.append(row)
        if len(self.demonstrations) > 4096:
            del self.demonstrations[: len(self.demonstrations) - 4096]
        if len(candidates) != 1:
            self.unresolved_demonstrations += 1
            return {"learned": False, "reason": "operation remains ambiguous", **row}
        op = candidates[0]
        w = max(1, int(weight))
        for feature in self._features(tokenized):
            counts = self.feature_counts.setdefault(feature, {name: 0 for name in OPERATIONS})
            counts[op] = int(counts.get(op, 0)) + w
        self.learned_turns += 1
        self._trim_features()
        return {"learned": True, "operation": op, "features": self._features(tokenized), **row}

    def teach_operator_phrase(self, phrase: str, operation: str, *, weight: int = 8) -> Dict[str, object]:
        op_name = self.normalize(operation)
        mapping = {
            "add": "ADD",
            "addition": "ADD",
            "sum": "ADD",
            "subtract": "SUB",
            "subtraction": "SUB",
            "minus": "SUB",
        }
        op = mapping.get(op_name, operation.upper())
        if op not in {"ADD", "SUB"}:
            return {"learned": False, "reason": f"unsupported operator meaning {operation!r}"}
        _, tokenized = self.extract_numbers(phrase)
        feats = self._features(tokenized)
        if not feats:
            return {"learned": False, "reason": "empty operator phrase"}
        w = max(1, int(weight))
        for feature in feats:
            counts = self.feature_counts.setdefault(feature, {name: 0 for name in OPERATIONS})
            counts[op] = int(counts.get(op, 0)) + w
        self.learned_turns += 1
        self._trim_features()
        return {"learned": True, "phrase": self.normalize(phrase), "operation": op, "features": feats}

    def _score(self, tokenized: Sequence[str]) -> Tuple[Dict[str, float], List[str]]:
        scores = {op: 0.0 for op in OPERATIONS}
        evidence: List[str] = []
        for feature in self._features(tokenized):
            counts = self.feature_counts.get(feature)
            if not counts:
                continue
            ordered = sorted(((int(counts.get(op, 0)), op) for op in OPERATIONS), reverse=True)
            best_count, best_op = ordered[0]
            second_count = ordered[1][0]
            discrimination = best_count - second_count
            if best_count <= 0 or discrimination <= 0:
                continue
            strength = math.log1p(discrimination) + 0.35 * math.log1p(best_count)
            scores[best_op] += strength
            evidence.append(f"{feature}->{best_op}:{best_count}/{second_count}")
        return scores, evidence

    def infer(self, surface: str) -> ArithmeticInterpretationV19:
        self.inference_turns += 1
        numbers, tokenized = self.extract_numbers(surface)
        if len(numbers) != 2:
            return ArithmeticInterpretationV19(False, "I need exactly two numeric operands for this arithmetic test.")
        a, b = numbers
        scores, evidence = self._score(tokenized)
        ranked = sorted(((score, op) for op, score in scores.items()), reverse=True)
        best_score, best_op = ranked[0]
        second_score = ranked[1][0]
        if best_score <= 0.0:
            return ArithmeticInterpretationV19(
                False,
                "I can see two numbers, but I have not learned enough English evidence to identify the requested operation.",
                a,
                b,
                scores=scores,
            )
        margin = best_score - second_score
        if margin < 0.40:
            return ArithmeticInterpretationV19(
                False,
                "The arithmetic wording is ambiguous in my learned construction memory.",
                a,
                b,
                confidence=min(0.74, 0.50 + margin * 0.20),
                evidence=tuple(evidence[-12:]),
                scores=scores,
            )
        result = _apply(best_op, a, b)
        confidence = min(0.995, 0.70 + 0.08 * margin)
        symbol = "+" if best_op == "ADD" else "-"
        left, right = (a, b) if best_op != "RSUB" else (b, a)
        text = f"{_fmt(left)} {symbol} {_fmt(right)} = {_fmt(result)}."
        return ArithmeticInterpretationV19(
            True,
            text,
            a,
            b,
            best_op,
            result,
            confidence,
            tuple(evidence[-12:]),
            scores,
        )

    def bootstrap_english(self) -> None:
        """Teach a compact seed curriculum through the same online API.

        These are examples, not parser branches. Held-out tests use new numbers,
        recombined wrappers, and newly invented operator words learned at runtime.
        """
        curriculum = (
            ("what is 2 plus 3", 5),
            ("how much is 6 plus 9", 15),
            ("add 4 and 7", 11),
            ("calculate the sum of 8 and 5", 13),
            ("2 + 9", 11),
            ("what is 9 minus 4", 5),
            ("how much is 14 minus 6", 8),
            ("calculate 12 minus 5", 7),
            ("subtract 3 from 10", 7),
            ("take 2 away from 11", 9),
            ("13 - 8", 5),
        )
        for surface, result in curriculum:
            row = self.observe(surface, result, source="bootstrap", weight=3)
            if row.get("learned"):
                self.bootstrap_count += 1

    def summary(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "feature_count": len(self.feature_counts),
            "demonstrations": len(self.demonstrations),
            "bootstrap_demonstrations": self.bootstrap_count,
            "learned_turns": self.learned_turns,
            "inference_turns": self.inference_turns,
            "unresolved_demonstrations": self.unresolved_demonstrations,
            "operations": list(OPERATIONS),
            "backpropagation": False,
            "gradient_descent": False,
            "external_model": False,
            "pretrained_language_model": False,
            "memorized_arithmetic_table": False,
        }

    def save(self, path: str | Path) -> None:
        data = {
            "version": self.VERSION,
            "max_features": self.max_features,
            "feature_counts": self.feature_counts,
            "demonstrations": self.demonstrations,
            "unresolved_demonstrations": self.unresolved_demonstrations,
            "learned_turns": self.learned_turns,
            "inference_turns": self.inference_turns,
            "bootstrap_count": self.bootstrap_count,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "OnlineConstructionArithmeticV19":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(max_features=int(data.get("max_features", 8192)), bootstrap=False)
        obj.feature_counts = {
            str(feature): {op: int(row.get(op, 0)) for op in OPERATIONS}
            for feature, row in data.get("feature_counts", {}).items()
        }
        obj.demonstrations = list(data.get("demonstrations", []))[-4096:]
        obj.unresolved_demonstrations = int(data.get("unresolved_demonstrations", 0))
        obj.learned_turns = int(data.get("learned_turns", 0))
        obj.inference_turns = int(data.get("inference_turns", 0))
        obj.bootstrap_count = int(data.get("bootstrap_count", 0))
        return obj
