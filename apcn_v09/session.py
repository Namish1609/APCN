from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import json

from .learner import SemanticLanguageLearner
from .semantic import SemanticNode
from .teacher import LanguageEpisode, Lexicon, SemanticTeacher
from .testing import SemanticTestReport, run_semantic_test


@dataclass
class LanguageStep:
    episode: LanguageEpisode
    prediction: Optional[SemanticNode]
    learned: bool


class SemanticSessionV09:
    def __init__(self, seed: int = 9, learner: Optional[SemanticLanguageLearner] = None, lexicon: Optional[Lexicon] = None):
        self.seed = seed
        self.teacher = SemanticTeacher(seed=seed, lexicon=lexicon)
        self.learner = learner or SemanticLanguageLearner()
        self.curriculum_index = self.learner.episode_count
        self.phase_counts: Dict[str, int] = {}
        self.history: List[Dict[str, float]] = []

    def step(self) -> LanguageStep:
        episodes = self.teacher.curriculum_episode(self.curriculum_index)
        last: Optional[LanguageStep] = None
        for ep in episodes:
            pred = self.learner.parse(ep.utterance, ep.discourse_focus)
            self.learner.observe(ep)
            self.curriculum_index += 1
            self.phase_counts[ep.phase] = self.phase_counts.get(ep.phase, 0) + 1
            last = LanguageStep(ep, pred, True)
        assert last is not None
        return last

    def train(self, n: int) -> LanguageStep:
        last = None
        for _ in range(max(1, int(n))):
            last = self.step()
        return last

    def preview(self, intent: Optional[str] = None, held_out: bool = False) -> LanguageStep:
        ep = self.teacher.simple(intent=intent, held_out=held_out, phase="preview")
        return LanguageStep(ep, self.learner.parse(ep.utterance, ep.discourse_focus), False)

    def test(self, samples: int = 300, held_out_templates: bool = False, seed: Optional[int] = None) -> SemanticTestReport:
        rep = run_semantic_test(self.learner, samples=samples, seed=seed or (self.seed + 9000 + len(self.history)), held_out_templates=held_out_templates)
        self.history.append({"episodes": float(self.learner.episode_count), "exact": rep.exact_accuracy, "intent": rep.intent_accuracy, "relation": rep.relation_accuracy, "operator": rep.operator_accuracy})
        return rep

    def save(self, output_dir: str | Path = "outputs/v0_9") -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        memory = out / "semantic_memory_v0_9.json"
        self.learner.save(memory)
        (out / "session_v0_9.json").write_text(json.dumps({"version": "0.9.0", "seed": self.seed, "phase_counts": self.phase_counts, "history": self.history}, indent=2), encoding="utf-8")
        return memory

    @classmethod
    def load(cls, path: str | Path, seed: int = 9) -> "SemanticSessionV09":
        return cls(seed=seed, learner=SemanticLanguageLearner.load(path))
