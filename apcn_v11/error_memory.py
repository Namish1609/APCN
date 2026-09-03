from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import math


@dataclass
class ErrorSignature:
    domain: str
    truth: str
    predicted: str
    context: str = ""
    count: int = 0
    recent_weight: float = 0.0

    @property
    def key(self) -> Tuple[str, str, str, str]:
        return (self.domain, self.truth, self.predicted, self.context)


class ErrorMemory:
    """Aggregate failure memory with bounded diagnostic representatives."""

    VERSION = "APCN-V0.11-ERROR-MEMORY"

    def __init__(self, decay: float = 0.997, representative_limit: int = 24):
        self.decay = float(decay)
        self.representative_limit = int(representative_limit)
        self.signatures: Dict[Tuple[str, str, str, str], ErrorSignature] = {}
        self.representatives: Dict[Tuple[str, str, str, str], List[str]] = {}
        self.observations = 0

    def _decay_domain(self, domain: str) -> None:
        for s in self.signatures.values():
            if s.domain == domain:
                s.recent_weight *= self.decay

    def record(self, domain: str, truth: str, predicted: str, *, context: str = "",
               representative: str = "") -> None:
        if truth == predicted:
            return
        self.observations += 1
        self._decay_domain(domain)
        key = (str(domain), str(truth), str(predicted), str(context))
        s = self.signatures.get(key)
        if s is None:
            s = ErrorSignature(*key)
            self.signatures[key] = s
        s.count += 1
        s.recent_weight += 1.0
        if representative:
            rows = self.representatives.setdefault(key, [])
            if representative not in rows and len(rows) < self.representative_limit:
                rows.append(representative)

    def record_visual_report(self, report) -> None:
        for f in getattr(report, "failures", []):
            tc = str(getattr(f, "truth_color", "")); pc = str(getattr(f, "pred_color", ""))
            ts = str(getattr(f, "truth_shape", "")); ps = str(getattr(f, "pred_shape", ""))
            if tc and pc and tc != pc:
                self.record("visual_color", tc, pc)
            if ts and ps and ts != ps:
                self.record("visual_shape", ts, ps)

    @staticmethod
    def _program_path(text: str) -> str:
        """Compact outer semantic path, e.g. NEGATE>ASSERT or GOAL."""
        lines = [x.strip() for x in str(text).splitlines() if x.strip()]
        if not lines:
            return "NONE"
        ops = {"ASSERT", "QUERY", "GOAL", "NEGATE", "GROUP", "SEQUENCE"}
        path = []
        for line in lines[:4]:
            head = line.split("(", 1)[0].strip()
            if head in ops:
                path.append(head)
            else:
                break
        return ">".join(path) if path else lines[0]

    def record_language_report(self, report) -> None:
        for f in getattr(report, "failures", []):
            expected = str(getattr(f, "expected", ""))
            predicted = str(getattr(f, "predicted", ""))
            utterance = str(getattr(f, "utterance", ""))
            skill = str(getattr(f, "skill", "language"))
            epath = self._program_path(expected)
            ppath = self._program_path(predicted)
            if epath != ppath:
                self.record("language_program", epath, ppath, context=skill, representative=utterance)
            elif expected.strip() != predicted.strip():
                # Outer operator was correct but some semantic content/identity
                # was not. Keep this separate from intent/operator mistakes.
                domain = "language_reference" if skill == "reference" else "language_semantics"
                truth = "correct_entity_identity" if skill == "reference" else "expected_semantics"
                pred = "wrong_or_missing_entity" if skill == "reference" else "mismatched_semantics"
                self.record(domain, truth, pred, context=skill, representative=utterance)

    def top(self, domain: Optional[str] = None, limit: int = 12) -> List[ErrorSignature]:
        rows = [s for s in self.signatures.values() if domain is None or s.domain == domain]
        rows.sort(key=lambda s: (s.recent_weight, math.log1p(s.count)), reverse=True)
        return rows[:limit]

    def total_weight(self, domain: Optional[str] = None) -> float:
        return float(sum(s.recent_weight for s in self.signatures.values()
                         if domain is None or s.domain == domain))

    def summary(self) -> Dict[str, object]:
        domains: Dict[str, int] = {}
        for s in self.signatures.values():
            domains[s.domain] = domains.get(s.domain, 0) + 1
        return {
            "observations": self.observations,
            "signature_count": len(self.signatures),
            "domains": domains,
            "top": [asdict(s) for s in self.top(limit=10)],
            "representative_examples_retained": sum(len(v) for v in self.representatives.values()),
        }

    def save(self, path: str | Path) -> None:
        data = {
            "version": self.VERSION,
            "decay": self.decay,
            "representative_limit": self.representative_limit,
            "observations": self.observations,
            "signatures": [asdict(s) for s in self.signatures.values()],
            "representatives": {"\t".join(k): v for k, v in self.representatives.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ErrorMemory":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(data.get("decay", .997), data.get("representative_limit", 24))
        obj.observations = int(data.get("observations", 0))
        for row in data.get("signatures", []):
            s = ErrorSignature(**row)
            obj.signatures[s.key] = s
        for key, rows in data.get("representatives", {}).items():
            parts = tuple(key.split("\t"))
            if len(parts) == 4:
                obj.representatives[parts] = list(rows)
        return obj
