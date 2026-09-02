from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apcn_v07.generator import ProceduralTeacher
from apcn_v07.learner import GroundedConceptLearner
from apcn_v07.curriculum import CurriculumEngine
from apcn_v07.trainer import evaluate


class APCNV07Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.teacher = ProceduralTeacher(seed=101)
        cls.learner = GroundedConceptLearner()
        cls.curriculum = CurriculumEngine(cls.teacher, cls.learner, seed=202)
        for _ in range(2200):
            ep, _ = cls.curriculum.next_episode()
            cls.learner.train_episode(ep)

    def test_words_are_not_hardwired_to_feature_names(self):
        feature_ids = self.learner.sensor.feature_ids()
        self.assertTrue(all(fid.startswith("f") for fid in feature_ids))
        self.assertNotIn("yellow", feature_ids)
        self.assertNotIn("circle", feature_ids)

    def test_color_word_selects_channel_subspace(self):
        mass = self.learner.diagnostic_group_mass("yellow")
        self.assertGreater(mass["channel_signal"], mass["geometry_signal"])
        self.assertGreater(mass["channel_signal"], 0.55)

    def test_shape_word_selects_geometry_subspace(self):
        mass = self.learner.diagnostic_group_mass("circle")
        self.assertGreater(mass["geometry_signal"], mass["channel_signal"])
        self.assertGreater(mass["geometry_signal"], 0.45)

    def test_function_word_is_not_strong_visual_concept(self):
        content = min(self.learner.concept_quality("yellow"), self.learner.concept_quality("circle"))
        function = max(self.learner.concept_quality("this"), self.learner.concept_quality("is"))
        self.assertGreater(content, function + 0.08)

    def test_held_out_generalization(self):
        report = evaluate(self.learner, ProceduralTeacher(seed=303), samples=240, difficulty=0.86)
        self.assertGreaterEqual(report.color_accuracy, 0.90)
        self.assertGreaterEqual(report.shape_accuracy, 0.75)
        self.assertGreaterEqual(report.joint_accuracy, 0.70)

    def test_compact_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "memory.json"
            self.learner.save(path)
            loaded = GroundedConceptLearner.load(path)
            self.assertEqual(loaded.episode_count, self.learner.episode_count)
            self.assertEqual(set(loaded.token_stats), set(self.learner.token_stats))
            ep = ProceduralTeacher(seed=404).generate(color="yellow", shape="circle", difficulty=0.7)
            x = loaded.sensor.extract(ep.image, ep.attention_mask)
            c, _ = loaded.best_of(self.teacher.color_words, x)
            s, _ = loaded.best_of(self.teacher.shape_words, x)
            self.assertEqual(c, "yellow")
            self.assertEqual(s, "circle")

    def test_memory_is_sufficient_statistics_not_episode_archive(self):
        yellow = self.learner.token_stats["yellow"]
        self.assertEqual(yellow.sum.shape[0], self.learner.sensor.dim)
        self.assertFalse(hasattr(yellow, "episodes"))
        self.assertFalse(hasattr(self.learner, "episode_archive"))


if __name__ == "__main__":
    unittest.main()
