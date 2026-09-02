from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List
import random

from apcn_v07.generator import ProceduralTeacher
from .learner_v082 import CalibratedConceptLearner


@dataclass
class FailureCase:
    truth_color: str
    truth_shape: str
    pred_color: str
    pred_shape: str
    color_score: float
    shape_score: float


@dataclass
class BulkTestReport:
    samples: int
    difficulty: float
    color_accuracy: float
    shape_accuracy: float
    joint_accuracy: float
    color_labels: List[str]
    shape_labels: List[str]
    color_confusion: List[List[int]]
    shape_confusion: List[List[int]]
    color_recall: Dict[str, float]
    shape_recall: Dict[str, float]
    failures: List[FailureCase]
    learner_episode_count_before: int
    learner_episode_count_after: int

    def to_dict(self) -> Dict[str, object]:
        out = asdict(self)
        out["failures"] = [asdict(x) for x in self.failures]
        return out


def _recall(labels: List[str], matrix: List[List[int]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, label in enumerate(labels):
        total = sum(matrix[i])
        out[label] = (matrix[i][i] / total) if total else 0.0
    return out


def run_bulk_test(
    learner: CalibratedConceptLearner,
    samples: int = 500,
    difficulty: float = 0.80,
    seed: int = 28082,
    keep_failures: int = 30,
) -> BulkTestReport:
    """Balanced prediction-only benchmark. This function never calls observe()."""
    samples = max(1, int(samples))
    difficulty = max(0.0, min(1.0, float(difficulty)))
    teacher = ProceduralTeacher(seed=seed)
    rng = random.Random(seed + 1)

    colors = list(teacher.color_words)
    shapes = list(teacher.shape_words)
    cidx = {v: i for i, v in enumerate(colors)}
    sidx = {v: i for i, v in enumerate(shapes)}
    cm_c = [[0 for _ in colors] for _ in colors]
    cm_s = [[0 for _ in shapes] for _ in shapes]

    before = learner.episode_count
    color_ok = shape_ok = joint_ok = 0
    failures: List[FailureCase] = []

    # Cycle through every color × shape combination before repeating so the
    # score is not distorted by a lucky random class distribution.
    combos = [(c, s) for c in colors for s in shapes]
    rng.shuffle(combos)

    for i in range(samples):
        if i and i % len(combos) == 0:
            rng.shuffle(combos)
        true_c, true_s = combos[i % len(combos)]
        ep = teacher.generate(
            color=true_c,
            shape=true_s,
            difficulty=difficulty,
            add_distractors=difficulty >= 0.35,
        )
        x = learner.sensor.extract(ep.image, ep.attention_mask)
        pred_c, cscore = learner.best_of(colors, x)
        pred_s, sscore = learner.best_of(shapes, x)
        pred_c = pred_c or "?"
        pred_s = pred_s or "?"

        if pred_c in cidx:
            cm_c[cidx[true_c]][cidx[pred_c]] += 1
        if pred_s in sidx:
            cm_s[sidx[true_s]][sidx[pred_s]] += 1

        cok = pred_c == true_c
        sok = pred_s == true_s
        color_ok += int(cok)
        shape_ok += int(sok)
        joint_ok += int(cok and sok)
        if (not cok or not sok) and len(failures) < keep_failures:
            failures.append(FailureCase(
                truth_color=true_c,
                truth_shape=true_s,
                pred_color=pred_c,
                pred_shape=pred_s,
                color_score=float(cscore),
                shape_score=float(sscore),
            ))

    after = learner.episode_count
    if after != before:
        raise RuntimeError("bulk test changed learner memory; evaluation must be read-only")

    return BulkTestReport(
        samples=samples,
        difficulty=difficulty,
        color_accuracy=color_ok / samples,
        shape_accuracy=shape_ok / samples,
        joint_accuracy=joint_ok / samples,
        color_labels=colors,
        shape_labels=shapes,
        color_confusion=cm_c,
        shape_confusion=cm_s,
        color_recall=_recall(colors, cm_c),
        shape_recall=_recall(shapes, cm_s),
        failures=failures,
        learner_episode_count_before=before,
        learner_episode_count_after=after,
    )
