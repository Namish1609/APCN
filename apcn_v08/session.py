from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import json
import numpy as np

from apcn_v07.generator import GroundedEpisode, ProceduralTeacher
from apcn_v07.curriculum import CurriculumEngine, CurriculumState
from apcn_v07.trainer import evaluate, EvaluationReport
from .learner import EnhancedConceptLearner


@dataclass
class StepResult:
    episode: GroundedEpisode
    state: CurriculumState
    features: np.ndarray
    predicted_color: Optional[str]
    color_score: float
    predicted_shape: Optional[str]
    shape_score: float


class TrainingSessionV08:
    def __init__(self, seed: int = 8, learner: Optional[EnhancedConceptLearner] = None):
        self.seed = seed
        self.teacher = ProceduralTeacher(seed=seed)
        self.learner = learner or EnhancedConceptLearner()
        self.curriculum = CurriculumEngine(self.teacher, self.learner, seed=seed + 101)
        self.curriculum.index = self.learner.episode_count
        self.phase_counts: Dict[str, int] = {}
        self.last_step: Optional[StepResult] = None

    def step(self) -> StepResult:
        ep, state = self.curriculum.next_episode()
        x = self.learner.train_episode(ep)
        self.phase_counts[state.phase] = self.phase_counts.get(state.phase, 0) + 1
        pc, pcs = self.learner.best_of(self.teacher.color_words, x)
        ps, pss = self.learner.best_of(self.teacher.shape_words, x)
        result = StepResult(ep, state, x, pc, pcs, ps, pss)
        self.last_step = result
        return result

    def generate_preview(
        self,
        color: Optional[str] = None,
        shape: Optional[str] = None,
        difficulty: float = 0.65,
        add_distractors: bool = True,
    ) -> StepResult:
        ep = self.teacher.generate(color=color, shape=shape, difficulty=difficulty, add_distractors=add_distractors)
        x = self.learner.sensor.extract(ep.image, ep.attention_mask)
        pc, pcs = self.learner.best_of(self.teacher.color_words, x)
        ps, pss = self.learner.best_of(self.teacher.shape_words, x)
        state = CurriculumState(self.learner.episode_count, "preview_only", difficulty, None)
        result = StepResult(ep, state, x, pc, pcs, ps, pss)
        self.last_step = result
        return result

    def teach_current(self, utterance: str, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        x = self.learner.sensor.extract(image, mask)
        self.learner.observe(utterance, x)
        self.curriculum.index = self.learner.episode_count
        return x

    def evaluate(self, samples: int = 250, difficulty: float = 0.86) -> EvaluationReport:
        return evaluate(self.learner, self.teacher, samples=samples, difficulty=difficulty)

    def save(self, output_dir: str | Path = "outputs/v0_8") -> Path:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        memory = output / "concept_memory_v0_8.json"
        self.learner.save(memory)
        (output / "session_v0_8.json").write_text(json.dumps({
            "seed": self.seed,
            "episode_count": self.learner.episode_count,
            "phase_counts": self.phase_counts,
            "memory_summary": self.learner.memory_summary(),
        }, indent=2), encoding="utf-8")
        return memory

    @classmethod
    def load(cls, path: str | Path, seed: int = 8) -> "TrainingSessionV08":
        learner = EnhancedConceptLearner.load(path)
        return cls(seed=seed, learner=learner)
