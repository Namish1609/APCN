from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import json

import numpy as np

from apcn_v10.language_common import LanguageEpisode
from apcn_v10.semantic import EntityRef, SemanticNode, semantic_equal
from apcn_v11.consolidation import ConsolidationEngine
from apcn_v13.session import CognitiveSessionV13
from .face import BBox, SelfFaceMemory
from .language import AdaptiveLanguageSessionV14, SemanticLanguageLearnerV14
from .language_teacher import RichSemanticTeacherV14


class CognitiveSessionV14(CognitiveSessionV13):
    VERSION = "0.14.0"

    def __init__(self, seed: int = 14):
        super().__init__(seed)
        self.seed = seed
        self.language = AdaptiveLanguageSessionV14(seed)
        self.self_face = SelfFaceMemory()
        # V0.14 deliberately stops treating perception as the only scaling
        # bottleneck. Most automatic learning budget now goes to language while
        # perception/world memory remains an active grounded test bed.
        self.language_budget_ratio = .80
        self.v14_language_history = []
        self.v14_face_history = []

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
        self.v14_language_history.append({"kind": "train", **result})
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

    def teach_language_paraphrase(self, utterance: str, program: SemanticNode,
                                  discourse_focus: Optional[EntityRef] = None) -> Dict[str, object]:
        """One explicit human demonstration with strong LOCAL construction weight.

        A repeated full `learner.observe()` on the same sentence/content would
        incorrectly make a new function word look like a color/shape/relation
        merely because it co-occurs with one scene. V0.14 therefore records the
        ordinary lexical experience ONCE, then strengthens only the aligned
        program-construction memory. This is analogous to a high-confidence
        correction, not global retraining.
        """
        text = str(utterance).strip()
        if not text:
            raise ValueError("paraphrase is empty")
        learner = self.language.learner
        before = learner.parse(text, discourse_focus, discourse_registry=self.language.discourse)
        episode = LanguageEpisode(text, program, "user_paraphrase", discourse_focus=discourse_focus)
        learner.observe(episode)  # one normal language experience only
        # One human demonstration is treated as stronger evidence than one
        # procedural exposure, but only inside the bounded construction store.
        for _ in range(3):
            learner.program_constructions.observe(learner, episode)
        after = learner.parse(text, discourse_focus, discourse_registry=self.language.discourse)
        row = {
            "utterance": text,
            "matched_before": semantic_equal(before, program),
            "matched_after": semantic_equal(after, program),
            "predicted_after": after.pretty() if after is not None else None,
            "construction_evidence": dict(learner.last_program_evidence),
        }
        self.v14_language_history.append({"kind": "explicit_paraphrase", **row})
        return row

    def test_rich_language(self, samples: int = 180, *, seed_offset: int = 1701) -> Dict[str, object]:
        """Read-only held-out construction test on the CURRENT language memory."""
        samples = max(1, int(samples))
        teacher = RichSemanticTeacherV14(self.seed + int(seed_offset))
        before = self.language.learner.episode_count
        exact = intent = relation = operator = 0
        failures = []
        for i in range(samples):
            ep = teacher.held_out_construction(i)
            pred = self.language.learner.parse(ep.utterance, discourse_registry=self.language.discourse)
            ok = semantic_equal(pred, ep.program)
            exact += int(ok)
            intent += int(pred is not None and pred.intent() == ep.program.intent())
            relation += int(pred is not None and pred.relations() == ep.program.relations())
            operator += int(pred is not None and pred.operators() == ep.program.operators())
            if not ok and len(failures) < 12:
                failures.append({
                    "utterance": ep.utterance,
                    "expected": ep.program.pretty(),
                    "predicted": pred.pretty() if pred is not None else None,
                    "construction_evidence": dict(self.language.learner.last_program_evidence),
                })
        after = self.language.learner.episode_count
        report = {
            "samples": samples,
            "exact": exact/samples,
            "intent": intent/samples,
            "relation": relation/samples,
            "operator": operator/samples,
            "memory_frozen": before == after,
            "episodes": after,
            "program_patterns": self.language.learner.program_constructions.summary(8),
            "failures": failures,
        }
        self.v14_language_history.append({"kind": "test", **{k:v for k,v in report.items() if k != "failures"}})
        return report

    # ---- opt-in local self-face API -------------------------------------------------
    # These methods deliberately support ONE user-enrolled local identity. They do
    # not perform public-person lookup or infer demographic attributes.
    def face_auto_bbox(self, frame: np.ndarray) -> Optional[BBox]:
        return self.self_face.auto_bbox(frame)

    def enroll_self_face(self, name: str, frame: np.ndarray, bbox: BBox) -> Dict[str, object]:
        row = self.self_face.enroll(name, frame, bbox)
        self.v14_face_history.append({"kind": "enroll", "name": row["name"], "views": row["views"]})
        return row

    def recognize_self_face(self, frame: np.ndarray, bbox: BBox) -> Dict[str, object]:
        row = self.self_face.recognize(frame, bbox)
        self.v14_face_history.append({"kind": "verify", "state": row.get("state"), "score": row.get("score", 0.0), "match": row.get("match", False)})
        return row

    def mark_face_not_me(self, frame: np.ndarray, bbox: BBox) -> Dict[str, object]:
        row = self.self_face.mark_not_me(frame, bbox)
        self.v14_face_history.append({"kind": "negative", **row})
        return row

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
            "v14_face_history": self.v14_face_history,
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
            data=json.loads(sp.read_text(encoding="utf-8"))
            obj.language_budget_ratio=float(data.get("language_budget_ratio",.80))
            obj.v14_language_history=list(data.get("v14_language_history",[]))
            obj.v14_face_history=list(data.get("v14_face_history",[]))
        obj.sync_graph(); return obj
