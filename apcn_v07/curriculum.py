from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence
import random

from .generator import GroundedEpisode, ProceduralTeacher
from .learner import GroundedConceptLearner


@dataclass
class CurriculumState:
    episode_index: int
    phase: str
    difficulty: float
    target_word: Optional[str]


class CurriculumEngine:
    """
    Automated teacher scheduling.

    1) factorial bootstrap breaks color/shape correlations;
    2) minimal contrasts reinforce factor separation;
    3) nuisance randomization removes position/size/lighting shortcuts;
    4) active phase chooses the weakest currently learned content word.
    """

    def __init__(self, teacher: ProceduralTeacher, learner: GroundedConceptLearner, seed: int = 13):
        self.teacher = teacher
        self.learner = learner
        self.rng = random.Random(seed)
        self.index = 0

    @property
    def content_words(self) -> List[str]:
        return self.teacher.color_words + self.teacher.shape_words

    def _phase(self, total_hint: int = 2000) -> str:
        if self.index < 350:
            return "factorial_bootstrap"
        if self.index < 800:
            return "minimal_contrast"
        if self.index < 1400:
            return "nuisance_randomization"
        return "active_learning"

    def _weakest_word(self) -> str:
        scored = [(self.learner.concept_quality(w), self.rng.random(), w) for w in self.content_words]
        scored.sort()
        return scored[0][2]

    def next_episode(self) -> tuple[GroundedEpisode, CurriculumState]:
        phase = self._phase()
        target_word: Optional[str] = None

        if phase == "factorial_bootstrap":
            ci = self.index % len(self.teacher.color_words)
            si = (self.index // len(self.teacher.color_words)) % len(self.teacher.shape_words)
            color = self.teacher.color_words[ci]
            shape = self.teacher.shape_words[si]
            difficulty = 0.15 + 0.15 * self.rng.random()

        elif phase == "minimal_contrast":
            if self.index % 2 == 0:
                color = self.rng.choice(self.teacher.color_words)
                shape = self.teacher.shape_words[(self.index // 2) % len(self.teacher.shape_words)]
                target_word = color
            else:
                shape = self.rng.choice(self.teacher.shape_words)
                color = self.teacher.color_words[(self.index // 2) % len(self.teacher.color_words)]
                target_word = shape
            difficulty = 0.25 + 0.15 * self.rng.random()

        elif phase == "nuisance_randomization":
            color = self.rng.choice(self.teacher.color_words)
            shape = self.rng.choice(self.teacher.shape_words)
            difficulty = 0.45 + 0.40 * self.rng.random()

        else:
            target_word = self._weakest_word()
            if target_word in self.teacher.color_words:
                color = target_word
                shape = self.rng.choice(self.teacher.shape_words)
            else:
                shape = target_word
                color = self.rng.choice(self.teacher.color_words)
            q = self.learner.concept_quality(target_word)
            difficulty = min(1.0, 0.58 + 0.42 * max(q, 0.15))

        episode = self.teacher.generate(color=color, shape=shape, difficulty=difficulty)
        state = CurriculumState(self.index, phase, difficulty, target_word)
        self.index += 1
        return episode, state
