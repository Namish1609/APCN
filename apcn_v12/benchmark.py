from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict

from apcn_v07.generator import ProceduralTeacher
from apcn_v11.visual import PrototypeConceptLearner
from .visual import SelfOrganizingVisualLearner


@dataclass
class VisualScore:
    samples: int
    color_accuracy: float
    shape_accuracy: float
    joint_accuracy: float
    shape_confusions: Dict[str, int]
    color_confusions: Dict[str, int]


@dataclass
class PairedBenchmarkReport:
    train_experiences: int
    test_samples: int
    difficulty: float
    v011: VisualScore
    v012: VisualScore

    def to_dict(self):
        return asdict(self)


def _score(learner, teacher: ProceduralTeacher, samples: int, difficulty: float) -> VisualScore:
    color_hit = shape_hit = joint_hit = 0
    sc = Counter(); cc = Counter()
    colors = teacher.color_words; shapes = teacher.shape_words
    for i in range(max(1, int(samples))):
        color = colors[i % len(colors)]
        shape = shapes[(i // len(colors)) % len(shapes)]
        ep = teacher.generate(color=color, shape=shape, difficulty=difficulty,
                              add_distractors=difficulty >= .4)
        x = learner.sensor.extract(ep.image, ep.attention_mask)
        pc, _ = learner.best_of(colors, x)
        ps, _ = learner.best_of(shapes, x)
        color_hit += int(pc == color); shape_hit += int(ps == shape)
        joint_hit += int(pc == color and ps == shape)
        if pc != color: cc[f"{color}->{pc}"] += 1
        if ps != shape: sc[f"{shape}->{ps}"] += 1
    n = max(1, int(samples))
    return VisualScore(n, color_hit/n, shape_hit/n, joint_hit/n,
                       dict(sc.most_common()), dict(cc.most_common()))


def run_paired_benchmark(train_experiences: int = 1800, test_samples: int = 600,
                         difficulty: float = .86, seed: int = 12012) -> PairedBenchmarkReport:
    """Train V0.11 and V0.12 on exactly the same grounded episodes, then test
    them on separate but identically distributed held-out scenes."""
    teacher = ProceduralTeacher(seed=seed)
    old = PrototypeConceptLearner()
    new = SelfOrganizingVisualLearner(representation_learning_until=min(1200, max(300, train_experiences)))
    colors = teacher.color_words; shapes = teacher.shape_words

    for i in range(max(0, int(train_experiences))):
        color = colors[i % len(colors)]
        shape = shapes[(i // len(colors)) % len(shapes)]
        d = .12 + .76 * ((i % 160) / 159.0)
        ep = teacher.generate(color=color, shape=shape, difficulty=d,
                              add_distractors=d >= .42)
        old.train_episode(ep)
        new.train_episode(ep)

    old_test = ProceduralTeacher(seed=seed + 991)
    new_test = ProceduralTeacher(seed=seed + 991)
    return PairedBenchmarkReport(
        int(train_experiences), int(test_samples), float(difficulty),
        _score(old, old_test, test_samples, difficulty),
        _score(new, new_test, test_samples, difficulty),
    )
