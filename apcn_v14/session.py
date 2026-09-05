from __future__ import annotations

from pathlib import Path
from typing import Dict
import json

from apcn_v11.consolidation import ConsolidationEngine
from apcn_v13.session import CognitiveSessionV13
from .face import SelfFaceMemory
from .language import AdaptiveLanguageSessionV14, SemanticLanguageLearnerV14


class CognitiveSessionV14(CognitiveSessionV13):
    VERSION = "0.14.0"

    def __init__(self, seed: int = 14):
        super().__init__(seed)
        self.seed = seed
        self.language = AdaptiveLanguageSessionV14(seed)
        self.self_face = SelfFaceMemory()
        self.language_budget_ratio = .80
        self.v14_language_history = []

    @staticmethod
    def _adopt_v13_state(obj: "CognitiveSessionV14", old: CognitiveSessionV13) -> None:
        obj.visual = old.visual
        obj.concepts = old.concepts
        obj.definitions = old.definitions
        obj.query = old.query
        obj.graph = old.graph
        obj.errors = old.errors
        obj.consolidation = ConsolidationEngine(obj.errors)
        obj.world = old.world
        obj.visual_test_history = list(old.visual_test_history)
        obj.language_test_history = list(old.language_test_history)
        obj.test_history = obj.language_test_history
        obj.consolidation_history = list(old.consolidation_history)
        obj.world_test_history = list(old.world_test_history)
        obj.v012_bootstrap_experiences = old.v012_bootstrap_experiences
        learner = SemanticLanguageLearnerV14.from_v12(old.language.learner)
        obj.language = AdaptiveLanguageSessionV14(obj.seed, learner=learner)
        obj.language.discourse = old.language.discourse

    @classmethod
    def from_v13_checkpoint(cls, output_dir: str | Path = "outputs/v0_13", *, seed: int = 14) -> "CognitiveSessionV14":
        old = CognitiveSessionV13.load_checkpoint(output_dir, seed=seed)
        obj = cls(seed)
        cls._adopt_v13_state(obj, old)
        return obj

    def language_first_train(self, steps: int = 400) -> Dict[str, object]:
        steps = max(1, int(steps))
        before = self.language.learner.episode_count
        correct_before = 0
        skills: Dict[str, int] = {}
        for _ in range(steps):
            row = self.language.step()
            correct_before += int(row.correct_before_learning)
            skills[row.skill] = skills.get(row.skill, 0) + 1
        after = self.language.learner.episode_count
        result = {
            "requested_steps": steps,
            "episodes_before": before,
            "episodes_after": after,
            "experiences_added": after-before,
            "correct_before_learning_rate": correct_before/steps,
            "skill_mix": skills,
            "program_constructions": self.language.learner.program_constructions.summary(8),
        }
        self.v14_language_history.append(result)
        return result

    def mixed_priority_train(self, total_steps: int = 500) -> Dict[str, object]:
        total_steps = max(2, int(total_steps))
        language_steps = max(1, int(round(total_steps*self.language_budget_ratio)))
        visual_steps = max(0, total_steps-language_steps)
        lang = self.language_first_train(language_steps)
        visual_added = 0
        if visual_steps:
            before = self.visual.learner.episode_count
            for _ in range(visual_steps):
                self.visual.step()
            visual_added = self.visual.learner.episode_count-before
        return {"total_steps":total_steps,"language_ratio":self.language_budget_ratio,
                "language":lang,"visual_experiences_added":visual_added}

    def memory_audit(self) -> Dict[str, object]:
        base = super().memory_audit()
        base["v014_language_program_memory"] = self.language.learner.program_constructions.summary(8)
        base["v014_self_face_memory"] = self.self_face.summary()
        base["v014_language_budget_ratio"] = self.language_budget_ratio
        return base

    def save(self, output_dir: str | Path = "outputs/v0_14") -> Dict[str, str]:
        if str(output_dir).replace("\\", "/").rstrip("/") == "outputs/v0_13":
            output_dir = "outputs/v0_14"
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        base_dir = out / "base_v13"
        super().save(base_dir)
        language_path = out / "language_memory_v0_14.json"; self.language.learner.save(language_path)
        discourse_path = out / "discourse_state_v0_14.json"; self.language.discourse.save(discourse_path)
        face_path = out / "self_face_memory_v0_14.json"; self.self_face.save(face_path)
        state_path = out / "session_v0_14.json"
        state_path.write_text(json.dumps({
            "version": self.VERSION,
            "seed": self.seed,
            "language_budget_ratio": self.language_budget_ratio,
            "v14_language_history": self.v14_language_history,
            "memory_audit": self.memory_audit(),
        }, indent=2), encoding="utf-8")
        return {"base_v13":str(base_dir),"language":str(language_path),"discourse":str(discourse_path),"self_face":str(face_path),"session":str(state_path)}

    @classmethod
    def load_checkpoint(cls, output_dir: str | Path = "outputs/v0_14", *, seed: int = 14) -> "CognitiveSessionV14":
        out = Path(output_dir); base_dir = out / "base_v13"
        if not base_dir.exists(): raise FileNotFoundError(f"missing V0.14 base checkpoint: {base_dir}")
        old = CognitiveSessionV13.load_checkpoint(base_dir, seed=seed)
        obj = cls(seed); cls._adopt_v13_state(obj, old)
        lp = out / "language_memory_v0_14.json"
        if lp.exists():
            obj.language = AdaptiveLanguageSessionV14(seed, learner=SemanticLanguageLearnerV14.load(lp))
            dp = out / "discourse_state_v0_14.json"
            if dp.exists():
                from apcn_v11.discourse import DiscourseEntityRegistry
                obj.language.discourse = DiscourseEntityRegistry.load(dp)
        fp = out / "self_face_memory_v0_14.json"
        if fp.exists(): obj.self_face = SelfFaceMemory.load(fp)
        sp = out / "session_v0_14.json"
        if sp.exists():
            data=json.loads(sp.read_text(encoding="utf-8")); obj.language_budget_ratio=float(data.get("language_budget_ratio",.80)); obj.v14_language_history=list(data.get("v14_language_history",[]))
        obj.sync_graph(); return obj
