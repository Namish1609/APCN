from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
import json
import numpy as np

from .generator import ProceduralTeacher
from .learner import GroundedConceptLearner
from .curriculum import CurriculumEngine


@dataclass
class EvaluationReport:
    episodes: int
    color_accuracy: float
    shape_accuracy: float
    joint_accuracy: float
    concept_profiles: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "episodes": self.episodes,
            "color_accuracy": self.color_accuracy,
            "shape_accuracy": self.shape_accuracy,
            "joint_accuracy": self.joint_accuracy,
            "concept_profiles": self.concept_profiles,
        }


def evaluate(
    learner: GroundedConceptLearner,
    teacher: ProceduralTeacher,
    samples: int = 300,
    difficulty: float = 0.80,
) -> EvaluationReport:
    color_ok = 0
    shape_ok = 0
    joint_ok = 0
    for _ in range(samples):
        ep = teacher.generate(difficulty=difficulty, add_distractors=True)
        x = learner.sensor.extract(ep.image, ep.attention_mask)
        pred_c, _ = learner.best_of(teacher.color_words, x)
        pred_s, _ = learner.best_of(teacher.shape_words, x)
        true_c = str(ep.teacher_metadata["color"])
        true_s = str(ep.teacher_metadata["shape"])
        c_ok = pred_c == true_c
        s_ok = pred_s == true_s
        color_ok += int(c_ok)
        shape_ok += int(s_ok)
        joint_ok += int(c_ok and s_ok)

    profiles = {}
    for word in teacher.color_words + teacher.shape_words:
        profiles[word] = {
            **learner.token_profile(word),
            "diagnostic_signal_mass": learner.diagnostic_group_mass(word),
        }
    for word in ("this", "is", "the", "a"):
        if word in learner.token_stats:
            profiles[word] = {
                **learner.token_profile(word),
                "diagnostic_signal_mass": learner.diagnostic_group_mass(word),
            }
    return EvaluationReport(
        episodes=learner.episode_count,
        color_accuracy=color_ok / float(samples),
        shape_accuracy=shape_ok / float(samples),
        joint_accuracy=joint_ok / float(samples),
        concept_profiles=profiles,
    )


def train(
    episodes: int,
    seed: int = 7,
    eval_samples: int = 300,
    output_dir: str | Path = "outputs/v0_7",
    progress_every: int = 250,
) -> tuple[GroundedConceptLearner, EvaluationReport]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher = ProceduralTeacher(seed=seed)
    learner = GroundedConceptLearner()
    curriculum = CurriculumEngine(teacher, learner, seed=seed + 101)

    phase_counts: Dict[str, int] = {}
    for i in range(episodes):
        ep, state = curriculum.next_episode()
        learner.train_episode(ep)
        phase_counts[state.phase] = phase_counts.get(state.phase, 0) + 1
        if progress_every and (i + 1) % progress_every == 0:
            yellow_q = learner.concept_quality("yellow")
            circle_q = learner.concept_quality("circle")
            print(
                f"[{i+1:5d}/{episodes}] phase={state.phase:<22} "
                f"quality(yellow)={yellow_q:.3f} quality(circle)={circle_q:.3f}"
            )

    report = evaluate(learner, teacher, samples=eval_samples, difficulty=0.82)
    learner.save(output_dir / "concept_memory_v0_7.json")
    payload = report.to_dict()
    payload["phase_counts"] = phase_counts
    (output_dir / "training_report_v0_7.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return learner, report
