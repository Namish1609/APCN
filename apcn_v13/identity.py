from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import uuid

import numpy as np


@dataclass
class AppearancePrototype:
    count: int
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def from_vector(cls, x: np.ndarray) -> "AppearancePrototype":
        x = np.asarray(x, dtype=np.float64)
        return cls(1, x.copy(), np.zeros_like(x))

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    def to_dict(self) -> Dict[str, object]:
        return {"count": self.count, "mean": self.mean.tolist(), "m2": self.m2.tolist()}

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "AppearancePrototype":
        return cls(
            int(data["count"]),
            np.asarray(data["mean"], dtype=np.float64),
            np.asarray(data["m2"], dtype=np.float64),
        )


class BoundedAppearanceBank:
    """Streaming multi-view appearance memory.

    The bank stores aggregate prototypes only. It never retains raw frames or
    individual descriptors. New modes are added until `max_prototypes`; after
    that the nearest aggregate prototype is updated.
    """

    def __init__(self, max_prototypes: int = 8, novelty_threshold: float = 0.105):
        self.max_prototypes = int(max_prototypes)
        self.novelty_threshold = float(novelty_threshold)
        self.prototypes: List[AppearancePrototype] = []
        self.observations = 0

    @staticmethod
    def distance(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        # Generic normalized RMS. The descriptor is mostly bounded [0,1], but
        # normalizing by local energy prevents sparse histogram blocks from
        # dominating merely because they contain many zeros.
        scale = max(0.08, float(np.sqrt(np.mean(a*a) + np.mean(b*b)) * 0.5))
        return float(np.sqrt(np.mean((a-b)**2)) / scale)

    def nearest(self, x: np.ndarray) -> Tuple[float, Optional[int]]:
        if not self.prototypes:
            return float("inf"), None
        rows = [(self.distance(p.mean, x), i) for i, p in enumerate(self.prototypes)]
        return min(rows)

    def observe(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=np.float64)
        self.observations += 1
        dist, idx = self.nearest(x)
        if idx is None:
            self.prototypes.append(AppearancePrototype.from_vector(x))
            return 0
        if dist > self.novelty_threshold and len(self.prototypes) < self.max_prototypes:
            self.prototypes.append(AppearancePrototype.from_vector(x))
            return len(self.prototypes) - 1
        self.prototypes[idx].update(x)
        return int(idx)

    def score(self, x: np.ndarray, temperature: float = 0.18) -> float:
        dist, _ = self.nearest(x)
        if not math.isfinite(dist):
            return 0.0
        return float(math.exp(-max(0.0, dist) / max(1e-6, temperature)))

    def to_dict(self) -> Dict[str, object]:
        return {
            "max_prototypes": self.max_prototypes,
            "novelty_threshold": self.novelty_threshold,
            "observations": self.observations,
            "prototypes": [p.to_dict() for p in self.prototypes],
            "raw_frames_retained": 0,
            "raw_descriptors_retained": 0,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "BoundedAppearanceBank":
        obj = cls(int(data.get("max_prototypes", 8)), float(data.get("novelty_threshold", .105)))
        obj.observations = int(data.get("observations", 0))
        obj.prototypes = [AppearancePrototype.from_dict(x) for x in data.get("prototypes", [])]
        if len(obj.prototypes) > obj.max_prototypes:
            raise ValueError("appearance bank exceeds configured prototype bound")
        return obj


@dataclass
class PersistentInstance:
    instance_id: str
    name: Optional[str] = None
    category: Tuple[str, ...] = ()
    positive: BoundedAppearanceBank = field(default_factory=BoundedAppearanceBank)
    negative: BoundedAppearanceBank = field(default_factory=lambda: BoundedAppearanceBank(max_prototypes=4))
    teaching_events: int = 0
    correction_events: int = 0

    def appearance_score(self, x: np.ndarray) -> float:
        positive = self.positive.score(x)
        negative = self.negative.score(x) if self.negative.prototypes else 0.0
        return float(np.clip(positive - 0.72 * negative, 0.0, 1.0))

    def to_dict(self) -> Dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "name": self.name,
            "category": list(self.category),
            "positive": self.positive.to_dict(),
            "negative": self.negative.to_dict(),
            "teaching_events": self.teaching_events,
            "correction_events": self.correction_events,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PersistentInstance":
        return cls(
            instance_id=str(data["instance_id"]),
            name=data.get("name"),
            category=tuple(str(x).lower() for x in data.get("category", [])),
            positive=BoundedAppearanceBank.from_dict(data.get("positive", {})),
            negative=BoundedAppearanceBank.from_dict(data.get("negative", {"max_prototypes": 4})),
            teaching_events=int(data.get("teaching_events", 0)),
            correction_events=int(data.get("correction_events", 0)),
        )


@dataclass(frozen=True)
class IdentityMatch:
    instance_id: Optional[str]
    name: Optional[str]
    score: float
    margin: float
    state: str
    appearance_score: float
    spatial_score: float


class InstanceMemory:
    """Explicit persistent object identity memory.

    Matching combines appearance, optional category compatibility and a spatial
    continuity prior from the world model. There is no gradient optimization.
    """

    def __init__(self, *, max_views: int = 8, strong_threshold: float = .58,
                 probable_threshold: float = .42, ambiguity_margin: float = .055):
        self.max_views = int(max_views)
        self.strong_threshold = float(strong_threshold)
        self.probable_threshold = float(probable_threshold)
        self.ambiguity_margin = float(ambiguity_margin)
        self.instances: Dict[str, PersistentInstance] = {}
        self.name_index: Dict[str, str] = {}

    def create(self, *, name: Optional[str] = None, category: Sequence[str] = ()) -> PersistentInstance:
        iid = f"obj_{uuid.uuid4().hex[:10]}"
        inst = PersistentInstance(
            instance_id=iid,
            name=name,
            category=tuple(str(x).lower() for x in category),
            positive=BoundedAppearanceBank(max_prototypes=self.max_views),
        )
        self.instances[iid] = inst
        if name:
            self.name_index[name.lower()] = iid
        return inst

    def by_name(self, name: str) -> Optional[PersistentInstance]:
        iid = self.name_index.get(str(name).lower())
        return self.instances.get(iid) if iid else None

    def teach(self, name: str, descriptor: np.ndarray, *, category: Sequence[str] = ()) -> PersistentInstance:
        inst = self.by_name(name)
        if inst is None:
            inst = self.create(name=name, category=category)
        elif category and not inst.category:
            inst.category = tuple(str(x).lower() for x in category)
        inst.positive.observe(descriptor)
        inst.teaching_events += 1
        return inst

    @staticmethod
    def _category_score(inst: PersistentInstance, category: Sequence[str]) -> float:
        observed = {str(x).lower() for x in category if str(x).strip()}
        known = set(inst.category)
        if not observed or not known:
            return .5
        overlap = len(observed & known)
        if overlap == 0:
            return 0.0
        return float(overlap / max(len(observed), len(known)))

    @staticmethod
    def _spatial_score(instance_id: str, center: Optional[Tuple[float, float]],
                       predicted_centers: Optional[Dict[str, Tuple[float, float]]]) -> float:
        if center is None or not predicted_centers or instance_id not in predicted_centers:
            return .5
        px, py = predicted_centers[instance_id]
        d = math.hypot(float(center[0]) - px, float(center[1]) - py)
        return float(math.exp(-d / .22))

    def match(self, descriptor: np.ndarray, *, category: Sequence[str] = (),
              center: Optional[Tuple[float, float]] = None,
              predicted_centers: Optional[Dict[str, Tuple[float, float]]] = None) -> IdentityMatch:
        rows = []
        for inst in self.instances.values():
            appearance = inst.appearance_score(descriptor)
            category_score = self._category_score(inst, category)
            spatial = self._spatial_score(inst.instance_id, center, predicted_centers)
            # Appearance remains dominant. Spatial continuity is useful only as
            # a prior and cannot rescue a visually incompatible object.
            score = appearance * (0.78 + .12 * category_score + .10 * spatial)
            rows.append((float(score), appearance, spatial, inst))
        rows.sort(key=lambda r: r[0], reverse=True)
        if not rows:
            return IdentityMatch(None, None, 0.0, 0.0, "NOVEL", 0.0, .5)
        best = rows[0]
        second = rows[1][0] if len(rows) > 1 else 0.0
        margin = best[0] - second
        if best[0] >= self.strong_threshold and margin >= self.ambiguity_margin:
            state = "KNOWN"
        elif best[0] >= self.probable_threshold and margin >= self.ambiguity_margin * .6:
            state = "PROBABLE"
        elif best[0] >= self.probable_threshold * .8:
            state = "AMBIGUOUS"
        else:
            state = "NOVEL"
        return IdentityMatch(best[3].instance_id if state != "NOVEL" else None,
                             best[3].name if state != "NOVEL" else None,
                             best[0], margin, state, best[1], best[2])

    def reinforce(self, instance_id: str, descriptor: np.ndarray) -> None:
        self.instances[instance_id].positive.observe(descriptor)

    def correct(self, *, wrong_instance_id: Optional[str], correct_name: str,
                descriptor: np.ndarray, category: Sequence[str] = ()) -> PersistentInstance:
        if wrong_instance_id and wrong_instance_id in self.instances:
            wrong = self.instances[wrong_instance_id]
            wrong.negative.observe(descriptor)
            wrong.correction_events += 1
        correct = self.by_name(correct_name)
        if correct is None:
            correct = self.create(name=correct_name, category=category)
        correct.positive.observe(descriptor)
        correct.correction_events += 1
        return correct

    def memory_summary(self) -> Dict[str, object]:
        positives = sum(len(x.positive.prototypes) for x in self.instances.values())
        negatives = sum(len(x.negative.prototypes) for x in self.instances.values())
        return {
            "instances": len(self.instances),
            "named_instances": len(self.name_index),
            "positive_prototypes": positives,
            "negative_prototypes": negatives,
            "max_views_per_instance": self.max_views,
            "raw_frames_retained": 0,
            "raw_descriptors_retained": 0,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "max_views": self.max_views,
            "strong_threshold": self.strong_threshold,
            "probable_threshold": self.probable_threshold,
            "ambiguity_margin": self.ambiguity_margin,
            "instances": {k: v.to_dict() for k, v in self.instances.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "InstanceMemory":
        obj = cls(
            max_views=int(data.get("max_views", 8)),
            strong_threshold=float(data.get("strong_threshold", .58)),
            probable_threshold=float(data.get("probable_threshold", .42)),
            ambiguity_margin=float(data.get("ambiguity_margin", .055)),
        )
        obj.instances = {k: PersistentInstance.from_dict(v) for k, v in data.get("instances", {}).items()}
        obj.name_index = {v.name.lower(): k for k, v in obj.instances.items() if v.name}
        return obj

    def save(self, path: str | Path) -> None:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "InstanceMemory":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
