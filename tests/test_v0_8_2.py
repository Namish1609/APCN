from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np

from apcn_v07.generator import ProceduralTeacher
from apcn_v08.learner_v082 import CalibratedConceptLearner
from apcn_v08.session import TrainingSessionV08
from apcn_v08.testing_v082 import run_bulk_test


class V082Tests(unittest.TestCase):
    def test_clean_preview_is_read_only(self):
        session = TrainingSessionV08(seed=82)
        before = session.learner.episode_count
        r = session.generate_preview(color="blue", shape="circle", difficulty=0.0, add_distractors=False)
        self.assertEqual(before, session.learner.episode_count)
        self.assertEqual(r.episode.teacher_metadata["color"], "blue")
        self.assertEqual(r.episode.teacher_metadata["shape"], "circle")
        self.assertGreater(np.count_nonzero(r.episode.attention_mask), 20)

    def test_bulk_test_does_not_change_memory(self):
        session = TrainingSessionV08(seed=83)
        for _ in range(180):
            session.step()
        before = session.learner.episode_count
        report = run_bulk_test(session.learner, samples=60, difficulty=0.55, seed=999)
        self.assertEqual(report.learner_episode_count_before, before)
        self.assertEqual(report.learner_episode_count_after, before)
        self.assertEqual(session.learner.episode_count, before)
        self.assertEqual(sum(map(sum, report.color_confusion)), 60)
        self.assertEqual(sum(map(sum, report.shape_confusion)), 60)

    def test_shared_candidate_relevance_is_nonzero_after_grounding(self):
        teacher = ProceduralTeacher(seed=84)
        learner = CalibratedConceptLearner()
        # Balanced small curriculum: each shape appears with several colors so
        # geometry can become discriminative independently of chromatic signal.
        for _ in range(8):
            for shape in teacher.shape_words:
                for color in teacher.color_words:
                    ep = teacher.generate(color=color, shape=shape, difficulty=0.15, add_distractors=False)
                    learner.train_episode(ep)
        rel = learner.candidate_relevance(teacher.shape_words)
        self.assertEqual(rel.shape, (learner.sensor.dim,))
        self.assertGreater(float(rel.sum()), 0.0)

    def test_v082_can_reload_its_memory(self):
        session = TrainingSessionV08(seed=85)
        for _ in range(80):
            session.step()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "memory.json"
            session.learner.save(path)
            loaded = CalibratedConceptLearner.load(path)
            self.assertEqual(loaded.episode_count, session.learner.episode_count)
            self.assertEqual(loaded.sensor.dim, session.learner.sensor.dim)
            self.assertIn("yellow", loaded.token_stats)


if __name__ == "__main__":
    unittest.main()
