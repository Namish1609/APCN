from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple
import json
import math
import numpy as np

from apcn_v07.sensor import AnonymousVisualSensor
from apcn_v07.stats import RunningStats
from .learner import EnhancedConceptLearner


class CalibratedConceptLearner(EnhancedConceptLearner):
    """V0.8.2 classifier with shared candidate-family calibration.

    The grounding memory is unchanged: each word still stores only running
    sufficient statistics over the anonymous 23-dimensional visual signal.

    V0.7/V0.8 compared candidate words with independently selected feature
    subsets. Those scores are not necessarily calibrated against each other;
    that particularly hurts similar geometric classes such as circle/square and
    ellipse/rectangle. V0.8.2 derives one *shared discriminative subspace* for
    all candidates in a comparison, then scores every candidate on exactly the
    same dimensions. This is a small classical-statistics/LDA-like operation,
    not a neural layer and not gradient training.
    """

    VERSION = "APCN-V0.8.2-CALIBRATED-CONCEPT-MEMORY"

    def candidate_relevance(self, candidates: Sequence[str]) -> np.ndarray:
        rows = []
        for token in candidates:
            stats = self.token_stats.get(token.lower())
            if stats is not None and stats.count >= 3:
                rows.append(stats)
        if len(rows) < 2:
            return np.zeros(self.sensor.dim, dtype=np.float64)

        means = np.stack([s.mean for s in rows], axis=0)
        variances = np.stack([s.var for s in rows], axis=0)
        between = np.var(means, axis=0)
        within = np.mean(variances, axis=0)
        global_var = self.global_stats.var

        # Dimensions that vary strongly *between* candidate concepts while being
        # relatively stable *inside* each concept are useful for discrimination.
        raw = between / (within + 0.08 * global_var + 1e-8)
        raw = np.log1p(np.maximum(raw, 0.0))
        raw[~np.isfinite(raw)] = 0.0
        return raw

    def best_of(self, candidates: Sequence[str], x: np.ndarray) -> Tuple[Optional[str], float]:
        names = [str(t).lower() for t in candidates]
        available = [
            t for t in names
            if t in self.token_stats and self.token_stats[t].count >= 3
        ]
        if len(available) < 2:
            return super().best_of(candidates, x)

        relevance = self.candidate_relevance(available)
        positive = relevance[relevance > 0]
        if positive.size == 0:
            return super().best_of(candidates, x)

        # Keep a sparse shared subspace. All candidates are then compared using
        # exactly the same weights, so scores are directly comparable.
        threshold = float(np.quantile(positive, 0.45))
        weights = np.where(relevance >= threshold, relevance, 0.0)
        if float(weights.sum()) <= 1e-12:
            weights = relevance
        weights = weights / max(float(weights.sum()), 1e-12)

        pooled_within = np.mean(
            np.stack([self.token_stats[t].var for t in available], axis=0), axis=0
        )
        scale2 = 0.55 * self.global_stats.var + 0.45 * pooled_within + 1e-6
        xv = np.asarray(x, dtype=np.float64)

        scored = []
        for token in names:
            stats = self.token_stats.get(token)
            if stats is None or stats.count < 3:
                scored.append((0.0, token))
                continue
            z2 = (xv - stats.mean) ** 2 / scale2
            dist = float(np.sum(weights * z2))
            quality = self.concept_quality(token)
            # Quality is only a mild reliability prior; distance dominates.
            reliability = 0.72 + 0.28 * quality
            score = math.exp(-0.5 * min(dist, 80.0)) * reliability
            scored.append((float(score), token))

        scored.sort(reverse=True)
        score, token = scored[0]
        return token, float(score)

    @classmethod
    def load(
        cls,
        path: str | Path,
        sensor: Optional[AnonymousVisualSensor] = None,
    ) -> "CalibratedConceptLearner":
        """Load V0.7/V0.8/V0.8.1 memory without discarding learned episodes."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(sensor=sensor)
        if int(data["feature_dim"]) != obj.sensor.dim:
            raise ValueError(
                f"saved feature dimension {data['feature_dim']} does not match "
                f"V0.8.2 sensor dimension {obj.sensor.dim}"
            )
        obj.episode_count = int(data.get("episode_count", 0))
        obj.global_stats = RunningStats.from_dict(data["global_stats"])
        obj.token_stats = {
            k: RunningStats.from_dict(v) for k, v in data["token_stats"].items()
        }
        return obj
