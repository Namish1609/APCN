from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import re
import numpy as np

from .sensor import AnonymousVisualSensor
from .stats import RunningStats
from .generator import GroundedEpisode


TOKEN_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")


@dataclass
class GroundingScore:
    token: str
    score: float
    quality: float
    support: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "token": self.token,
            "score": self.score,
            "quality": self.quality,
            "support": self.support,
        }


class GroundedConceptLearner:
    """
    Online cross-situational grounded-word learner.

    It keeps only sufficient statistics for each token, not all episodes. If a
    word repeatedly occurs while one sensory subspace stays stable and other
    dimensions vary, those stable/discriminative dimensions acquire high weight.
    """

    VERSION = "APCN-V0.7-CONCEPT-MEMORY"

    def __init__(self, sensor: Optional[AnonymousVisualSensor] = None):
        self.sensor = sensor or AnonymousVisualSensor()
        self.global_stats = RunningStats.empty(self.sensor.dim)
        self.token_stats: Dict[str, RunningStats] = {}
        self.episode_count = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]

    def observe(self, utterance: str, feature_vector: np.ndarray) -> None:
        x = np.asarray(feature_vector, dtype=np.float64)
        self.global_stats.update(x)
        for token in sorted(set(self.tokenize(utterance))):
            stats = self.token_stats.setdefault(token, RunningStats.empty(self.sensor.dim))
            stats.update(x)
        self.episode_count += 1

    def train_episode(self, episode: GroundedEpisode) -> np.ndarray:
        x = self.sensor.extract(episode.image, episode.attention_mask)
        self.observe(episode.utterance, x)
        return x

    def relevance(self, token: str) -> np.ndarray:
        token = token.lower()
        pos = self.token_stats.get(token)
        if pos is None or pos.count < 3 or self.global_stats.count <= pos.count + 2:
            return np.zeros(self.sensor.dim, dtype=np.float64)
        neg = pos.complement(self.global_stats)
        if neg.count < 3:
            return np.zeros(self.sensor.dim, dtype=np.float64)

        pvar = pos.var
        nvar = neg.var
        gvar = self.global_stats.var
        diff2 = (pos.mean - neg.mean) ** 2

        fisher = diff2 / (pvar + nvar + 1e-8)
        invariance = gvar / (pvar + 0.18 * gvar + 1e-8)
        raw = np.log1p(np.maximum(fisher * invariance, 0.0))
        raw[~np.isfinite(raw)] = 0.0
        return raw

    def concept_quality(self, token: str) -> float:
        stats = self.token_stats.get(token.lower())
        if stats is None:
            return 0.0
        rel = self.relevance(token)
        if not np.any(rel > 0):
            return 0.0
        top = np.sort(rel)[-min(5, len(rel)):]
        support = 1.0 - math.exp(-stats.count / 18.0)
        signal = float(np.mean(top))
        return float(np.clip((1.0 - math.exp(-signal)) * support, 0.0, 1.0))

    def similarity(self, token: str, x: np.ndarray) -> float:
        token = token.lower()
        pos = self.token_stats.get(token)
        if pos is None or pos.count < 3:
            return 0.0
        rel = self.relevance(token)
        if rel.sum() <= 1e-10:
            return 0.0
        threshold = np.quantile(rel[rel > 0], 0.55) if np.any(rel > 0) else 0.0
        weights = np.where(rel >= threshold, rel, 0.0)
        if weights.sum() <= 1e-10:
            weights = rel
        weights = weights / weights.sum()

        scale = np.sqrt(self.global_stats.var + 1e-5)
        z2 = ((np.asarray(x) - pos.mean) / scale) ** 2
        dist = float(np.sum(weights * z2))
        quality = self.concept_quality(token)
        return float(math.exp(-0.5 * dist) * quality)

    def ground_image(
        self,
        image: np.ndarray,
        attention_mask: np.ndarray,
        top_k: int = 8,
        min_quality: float = 0.08,
    ) -> List[GroundingScore]:
        x = self.sensor.extract(image, attention_mask)
        out: List[GroundingScore] = []
        for token, stats in self.token_stats.items():
            q = self.concept_quality(token)
            if q < min_quality:
                continue
            out.append(GroundingScore(token, self.similarity(token, x), q, stats.count))
        out.sort(key=lambda g: (g.score, g.quality, g.support), reverse=True)
        return out[:top_k]

    def best_of(self, candidates: Sequence[str], x: np.ndarray) -> Tuple[Optional[str], float]:
        scored = [(self.similarity(token, x), token) for token in candidates]
        if not scored:
            return None, 0.0
        scored.sort(reverse=True)
        score, token = scored[0]
        return token, float(score)

    def token_profile(self, token: str, top_k: int = 8) -> Dict[str, object]:
        stats = self.token_stats.get(token.lower())
        if stats is None:
            return {"token": token.lower(), "known": False}
        rel = self.relevance(token)
        idx = np.argsort(rel)[::-1][:top_k]
        return {
            "token": token.lower(),
            "known": True,
            "support": stats.count,
            "quality": self.concept_quality(token),
            "top_dimensions": [
                {"feature": f"f{i:03d}", "relevance": float(rel[i])}
                for i in idx if rel[i] > 0
            ],
        }

    def diagnostic_group_mass(self, token: str) -> Dict[str, float]:
        rel = self.relevance(token)
        total = float(rel.sum())
        if total <= 1e-12:
            return {name: 0.0 for name in self.sensor.layout.groups}
        return {
            name: float(rel[a:b].sum() / total)
            for name, (a, b) in self.sensor.layout.groups.items()
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "version": self.VERSION,
            "feature_dim": self.sensor.dim,
            "episode_count": self.episode_count,
            "global_stats": self.global_stats.to_dict(),
            "token_stats": {k: v.to_dict() for k, v in sorted(self.token_stats.items())},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, sensor: Optional[AnonymousVisualSensor] = None) -> "GroundedConceptLearner":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(sensor=sensor)
        if int(data["feature_dim"]) != obj.sensor.dim:
            raise ValueError("saved feature dimension does not match sensor")
        obj.episode_count = int(data.get("episode_count", 0))
        obj.global_stats = RunningStats.from_dict(data["global_stats"])
        obj.token_stats = {k: RunningStats.from_dict(v) for k, v in data["token_stats"].items()}
        return obj
