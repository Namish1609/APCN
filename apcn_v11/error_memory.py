from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
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
    """Aggregate failure memory.

    It intentionally does not archive every failed image/sentence. Repeated
    mistakes are collapsed into signatures such as rectangle->ellipse or
    ASSERT->GOAL. A small bounded set of representative strings may be kept for
    diagnostics, but learning priority is derived from aggregate evidence.
    """

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

    def record_language_report(self, report) -> None:
        for f in getattr(report, "failures", []):
            expected = str(getattr(f, "expected", ""))
            predicted = str(getattr(f, "predicted", ""))
            utterance = str(getattr(f, "utterance", ""))
            skill = str(getattr(f, "skill", "language"))
            # First program line carries ASSERT/QUERY/GOAL/NEGATE/GROUP/etc.
            e0 = expected.strip().splitlines()[0] if expected.strip() else "NONE"
            p0 = predicted.strip().splitlines()[0] if predicted.strip() else "NONE"
            self.record("language_program", e0, p0, context=skill, representative=utterance)

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
