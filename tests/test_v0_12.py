import tempfile
import unittest
from pathlib import Path

import numpy as np

from apcn_v07.generator import ProceduralTeacher
from apcn_v11.visual import PrototypeConceptLearner
from apcn_v12.sensor import SelfOrganizingPatchSensor
from apcn_v12.visual import SelfOrganizingVisualLearner
from apcn_v12.language import AdaptiveConstructionCalibrator
from apcn_v12.session import CognitiveSessionV12


class TestV012(unittest.TestCase):
    def test_sensor_replaces_23d_summary_with_generic_groups(self):
        s = SelfOrganizingPatchSensor(max_codewords=12)
        self.assertNotEqual(s.dim, 23)
        self.assertIn("pixel_distribution", s.layout.groups)
        self.assertIn("normalized_raster", s.layout.groups)
        self.assertIn("learned_patch_codebook", s.layout.groups)
        self.assertFalse(hasattr(s, "hu_moments"))
        self.assertFalse(hasattr(s, "circularity"))

    def test_codebook_is_bounded_and_extract_is_read_only(self):
        teacher = ProceduralTeacher(seed=1201)
        s = SelfOrganizingPatchSensor(max_codewords=10, novelty_threshold=.16)
        for _ in range(35):
            ep = teacher.generate(difficulty=.62, add_distractors=True)
            s.learn(ep.image, ep.attention_mask)
        self.assertGreater(len(s.codewords), 2)
        self.assertLessEqual(len(s.codewords), 10)
        before_counts = list(s.codeword_counts)
        before_words = [x.copy() for x in s.codewords]
        ep = teacher.generate(difficulty=.8, add_distractors=True)
        x = s.extract(ep.image, ep.attention_mask)
        self.assertEqual(x.shape, (s.dim,))
        self.assertEqual(before_counts, s.codeword_counts)
        for a, b in zip(before_words, s.codewords):
            self.assertTrue(np.array_equal(a, b))
        self.assertEqual(s.memory_summary()["raw_patches_retained"], 0)

    def test_visual_memory_remains_bounded_and_persistent(self):
        teacher = ProceduralTeacher(seed=1202)
        learner = SelfOrganizingVisualLearner(
            sensor=SelfOrganizingPatchSensor(max_codewords=8), max_prototypes=4,
            representation_learning_until=120)
        for i in range(120):
            ep = teacher.generate(
                color=teacher.color_words[i % len(teacher.color_words)],
                shape=teacher.shape_words[(i // len(teacher.color_words)) % len(teacher.shape_words)],
                difficulty=.45,
            )
            learner.train_episode(ep)
        self.assertEqual(learner.episode_count, 120)
        self.assertLessEqual(len(learner.sensor.codewords), 8)
        self.assertTrue(all(len(v) <= 4 for v in learner.prototype_banks.values()))
        self.assertFalse(hasattr(learner, "episodes"))
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "v12.json"
            learner.save(p)
            restored = SelfOrganizingVisualLearner.load(p)
        self.assertEqual(restored.episode_count, learner.episode_count)
        self.assertEqual(len(restored.sensor.codewords), len(learner.sensor.codewords))
        self.assertEqual(restored.sensor.dim, learner.sensor.dim)

    def test_migration_keeps_v11_visual_count_without_mixing_feature_spaces(self):
        teacher = ProceduralTeacher(seed=1203)
        old = PrototypeConceptLearner()
        for _ in range(24):
            old.train_episode(teacher.generate(difficulty=.3))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old.save(root / "visual_memory_v0_11.json")
            migrated = CognitiveSessionV12.from_v11_checkpoint(
                root, seed=12, representation_bootstrap=12)
        self.assertEqual(migrated.visual.learner.legacy_visual_episode_count, 24)
        self.assertEqual(migrated.visual.learner.episode_count, 0)
        self.assertGreater(len(migrated.visual.learner.sensor.codewords), 0)
        self.assertNotEqual(migrated.visual.learner.sensor.dim, old.sensor.dim)

    def test_first_consolidation_bootstraps_all_visual_factors(self):
        s = CognitiveSessionV12(seed=1204)
        self.assertFalse(s.visual_coverage()["complete"])
        result = s.consolidate_visual(180)
        self.assertEqual(result["trained"], 180)
        self.assertTrue(s.visual_coverage()["complete"])

    def test_moderate_grounded_visual_learning_is_above_chance(self):
        s = CognitiveSessionV12(seed=1205)
        s.visual.learner.representation_learning_until = 360
        s._balanced_visual_bootstrap(420)
        rep = s.test_visual(samples=90, difficulty=.45)
        self.assertGreater(rep.color_accuracy, .55)
        self.assertGreater(rep.shape_accuracy, .35)
        self.assertGreater(rep.joint_accuracy, .22)

    def test_recent_construction_correction_can_overcome_stale_evidence(self):
        c = AdaptiveConstructionCalibrator(decay=.86, max_patterns=64)
        pattern = "it is the case that <ENTITY> is <REL> the <ENTITY>"
        for _ in range(80):
            c.observe(pattern, "GOAL")
        label, _, _ = c.predict(pattern)
        self.assertEqual(label, "GOAL")
        for _ in range(28):
            c.observe(pattern, "ASSERT")
        label, conf, _ = c.predict(pattern)
        self.assertEqual(label, "ASSERT")
        self.assertGreater(conf, .5)
        self.assertLessEqual(c.summary()["patterns"], 64)

    def test_v012_ui_keeps_existing_layout_and_exposes_representation_status(self):
        text = Path("apcn_v12/ui.py").read_text(encoding="utf-8")
        self.assertIn("Self-Organizing Perception Studio", text)
        self.assertIn("Re-import V0.11 Knowledge", text)
        self.assertIn("patch codewords", text)
        self.assertNotIn("addTab(self._representation", text)


if __name__ == "__main__":
    unittest.main()
