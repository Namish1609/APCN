from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import json

import numpy as np

from apcn_v07.stats import RunningStats
from apcn_v11.visual import PrototypeConceptLearner, OnlinePrototype
from .sensor import SelfOrganizingPatchSensor


class SelfOrganizingVisualLearner(PrototypeConceptLearner):
    """V0.12 visual learner.

    Concept learning remains sparse sufficient statistics + bounded multimodal
    prototypes. The major change is upstream: descriptors now come from raw
    pixel distributions, normalized occupancy and an unlabeled competitive local
    patch codebook instead of the old 23 handcrafted summary dimensions.
    """

    VERSION = "APCN-V0.12-SELF-ORGANIZING-VISUAL-MEMORY"

    def __init__(self, sensor: Optional[SelfOrganizingPatchSensor] = None, *,
                 max_prototypes: int = 8, new_prototype_distance: float = 2.65,
                 representation_learning_until: int = 1600):
        sensor = sensor or SelfOrganizingPatchSensor()
        super().__init__(sensor=sensor, max_prototypes=max_prototypes,
                         new_prototype_distance=new_prototype_distance)
        self.representation_learning_until = int(representation_learning_until)
        self.legacy_visual_episode_count = 0

    @property
    def patch_sensor(self) -> SelfOrganizingPatchSensor:
        return self.sensor

    def train_episode(self, episode):
        # Local codebook learning is label-free. It sees only pixels + attention.
        if self.episode_count < self.representation_learning_until:
            self.patch_sensor.learn(episode.image, episode.attention_mask)
        x = self.patch_sensor.extract(episode.image, episode.attention_mask)
        self.observe(episode.utterance, x)
        return x

    def bootstrap_representation(self, teacher, experiences: int = 240,
                                 difficulty: float = .72) -> int:
        """Learn patch vocabulary without updating word/concept statistics."""
        done = 0
        for i in range(max(0, int(experiences))):
            d = min(1.0, max(.08, difficulty * (.35 + .65*((i % 80)/79.0))))
            ep = teacher.generate(difficulty=d, add_distractors=d >= .45)
            self.patch_sensor.learn(ep.image, ep.attention_mask)
            done += 1
        return done

    def representation_summary(self) -> Dict[str, object]:
        return {
            "sensor": self.patch_sensor.memory_summary(),
            "concept_episodes_v012": self.episode_count,
            "legacy_visual_episodes_not_replayed": self.legacy_visual_episode_count,
            "representation_learning_until": self.representation_learning_until,
            "concept_prototypes": self.prototype_summary(),
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "version": self.VERSION,
            "feature_dim": self.sensor.dim,
            "episode_count": self.episode_count,
            "legacy_visual_episode_count": self.legacy_visual_episode_count,
            "global_stats": self.global_stats.to_dict(),
            "token_stats": {k: v.to_dict() for k, v in sorted(self.token_stats.items())},
            "max_prototypes": self.max_prototypes,
            "new_prototype_distance": self.new_prototype_distance,
            "representation_learning_until": self.representation_learning_until,
            "prototype_banks": {k: [p.to_dict() for p in v]
                                for k, v in self.prototype_banks.items()},
            "sensor": self.patch_sensor.to_dict(),
        }
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SelfOrganizingVisualLearner":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        sensor = SelfOrganizingPatchSensor.from_dict(data["sensor"])
        obj = cls(
            sensor=sensor,
            max_prototypes=int(data.get("max_prototypes", 8)),
            new_prototype_distance=float(data.get("new_prototype_distance", 2.65)),
            representation_learning_until=int(data.get("representation_learning_until", 1600)),
        )
        if int(data.get("feature_dim", sensor.dim)) != sensor.dim:
            raise ValueError("saved V0.12 feature dimension does not match sensor")
        obj.episode_count = int(data.get("episode_count", 0))
        obj.legacy_visual_episode_count = int(data.get("legacy_visual_episode_count", 0))
        obj.global_stats = RunningStats.from_dict(data["global_stats"])
        obj.token_stats = {k: RunningStats.from_dict(v) for k, v in data.get("token_stats", {}).items()}
        obj.prototype_banks = {
            k: [OnlinePrototype.from_dict(row) for row in rows]
            for k, rows in data.get("prototype_banks", {}).items()
        }
        return obj
