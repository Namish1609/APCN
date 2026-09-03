from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import math
import numpy as np

from apcn_v08.learner_v082 import CalibratedConceptLearner
from apcn_v07.stats import RunningStats
from apcn_v07.sensor import AnonymousVisualSensor


@dataclass
class OnlinePrototype:
    count: int
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def from_vector(cls, x: np.ndarray) -> "OnlinePrototype":
        x = np.asarray(x, dtype=np.float64)
        return cls(1, x.copy(), np.zeros_like(x))

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    @property
    def var(self) -> np.ndarray:
        if self.count <= 1:
            return np.ones_like(self.mean) * 1e-5
        return np.maximum(self.m2 / (self.count - 1), 1e-8)

    def to_dict(self):
        return {"count": self.count, "mean": self.mean.tolist(), "m2": self.m2.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(int(d["count"]), np.asarray(d["mean"], dtype=np.float64), np.asarray(d["m2"], dtype=np.float64))


class PrototypeConceptLearner(CalibratedConceptLearner):
    """Compact multi-modal perceptual concepts.

    V0.7/0.8 stores one sufficient-statistics distribution per word. V0.11 adds
    at most `max_prototypes` streaming prototypes per token so nuisance modes do
    not all collapse into one centroid. Prototypes are aggregate clusters, not
    retained training examples.
    """

    VERSION = "APCN-V0.11-PROTOTYPE-CONCEPT-MEMORY"

    def __init__(self, sensor: Optional[AnonymousVisualSensor] = None, *, max_prototypes: int = 6,
                 new_prototype_distance: float = 2.45):
        super().__init__(sensor=sensor)
        self.max_prototypes = int(max_prototypes)
        self.new_prototype_distance = float(new_prototype_distance)
        self.prototype_banks: Dict[str, List[OnlinePrototype]] = {}

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        scale2 = self.global_stats.var + 1e-5
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2 / scale2)))

    def _update_bank(self, token: str, x: np.ndarray) -> None:
        bank = self.prototype_banks.setdefault(token, [])
        if not bank:
            bank.append(OnlinePrototype.from_vector(x))
            return
        rows = [(self._distance(p.mean, x), i) for i, p in enumerate(bank)]
        dist, idx = min(rows)
        if dist > self.new_prototype_distance and len(bank) < self.max_prototypes:
            bank.append(OnlinePrototype.from_vector(x))
        else:
            bank[idx].update(x)

    def observe(self, utterance: str, feature_vector: np.ndarray) -> None:
        x = np.asarray(feature_vector, dtype=np.float64)
        tokens = sorted(set(self.tokenize(utterance)))
        super().observe(utterance, x)
        for token in tokens:
            self._update_bank(token, x)

    def prototype_similarity(self, token: str, x: np.ndarray) -> float:
        bank = self.prototype_banks.get(token.lower(), [])
        if not bank:
            return 0.0
        relevance = self.relevance(token)
        if not np.any(relevance > 0):
            return 0.0
        positive = relevance[relevance > 0]
        threshold = float(np.quantile(positive, .50)) if positive.size else 0.0
        weights = np.where(relevance >= threshold, relevance, 0.0)
        weights /= max(float(weights.sum()), 1e-12)
        scale2 = self.global_stats.var + 1e-5
        vals = []
        for p in bank:
            d2 = float(np.sum(weights * ((np.asarray(x) - p.mean) ** 2 / scale2)))
            support = 1.0 - math.exp(-p.count / 8.0)
            vals.append(math.exp(-0.5 * min(d2, 80.0)) * (0.65 + 0.35 * support))
        return float(max(vals, default=0.0) * (0.72 + 0.28 * self.concept_quality(token)))

    def best_of(self, candidates: Sequence[str], x: np.ndarray) -> Tuple[Optional[str], float]:
        base_token, base_score = super().best_of(candidates, x)
        rows = []
        for token in [str(c).lower() for c in candidates]:
            ps = self.prototype_similarity(token, x)
            # Existing calibrated score remains a stabilizer while prototype
            # evidence learns enough support.
            bs = super().best_of([token], x)[1] if token in self.token_stats else 0.0
            bank_support = sum(p.count for p in self.prototype_banks.get(token, []))
            alpha = min(0.72, 0.18 + bank_support / 120.0)
            rows.append(((1.0 - alpha) * bs + alpha * ps, token))
        rows.sort(reverse=True)
        if rows and rows[0][0] > 0:
            return rows[0][1], float(rows[0][0])
        return base_token, base_score

    def prototype_summary(self) -> Dict[str, object]:
        return {
            "max_prototypes_per_token": self.max_prototypes,
            "tokens_with_prototypes": len(self.prototype_banks),
            "total_prototypes": sum(len(v) for v in self.prototype_banks.values()),
            "per_token": {k: [p.count for p in v] for k, v in sorted(self.prototype_banks.items())},
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "version": self.VERSION,
            "feature_dim": self.sensor.dim,
            "episode_count": self.episode_count,
            "global_stats": self.global_stats.to_dict(),
            "token_stats": {k: v.to_dict() for k, v in sorted(self.token_stats.items())},
            "max_prototypes": self.max_prototypes,
            "new_prototype_distance": self.new_prototype_distance,
            "prototype_banks": {k: [p.to_dict() for p in v] for k, v in self.prototype_banks.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, sensor: Optional[AnonymousVisualSensor] = None) -> "PrototypeConceptLearner":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(sensor=sensor, max_prototypes=int(data.get("max_prototypes", 6)),
                  new_prototype_distance=float(data.get("new_prototype_distance", 2.45)))
        if int(data["feature_dim"]) != obj.sensor.dim:
            raise ValueError("saved feature dimension does not match V0.11 sensor dimension")
        obj.episode_count = int(data.get("episode_count", 0))
        obj.global_stats = RunningStats.from_dict(data["global_stats"])
        obj.token_stats = {k: RunningStats.from_dict(v) for k, v in data["token_stats"].items()}
        obj.prototype_banks = {
            k: [OnlinePrototype.from_dict(row) for row in rows]
            for k, rows in data.get("prototype_banks", {}).items()
        }
        return obj
