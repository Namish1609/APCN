import tempfile
import unittest
from pathlib import Path

import numpy as np

from apcn_v10.semantic import semantic_equal
from apcn_v14.face import SelfFaceMemory
from apcn_v14.face_benchmark import SyntheticSelfFaceTeacher, run_face_benchmark
from apcn_v14.language import ProgramConstructionMemory
from apcn_v14.language_teacher import RichSemanticTeacherV14
from apcn_v14.session import CognitiveSessionV14


class TestV014(unittest.TestCase):
    def test_self_face_memory_is_local_bounded_and_non_neural(self):
        teacher = SyntheticSelfFaceTeacher(1401)
        mem = SelfFaceMemory(max_views=6)
        for i in range(12):
            frame, bbox = teacher.render(1511, 1600+i)
            mem.enroll("me", frame, bbox)
        summary = mem.summary()
        self.assertTrue(summary["single_identity"])
        self.assertFalse(summary["neural_face_encoder"])
        self.assertEqual(summary["raw_face_images_retained"], 0)
        self.assertEqual(summary["raw_camera_frames_retained"], 0)
        self.assertLessEqual(summary["instance_memory"]["positive_prototypes"], 6)
        self.assertEqual(summary["locator"]["identity_role"], False)

    def test_self_face_negative_correction_reduces_wrong_score(self):
        teacher = SyntheticSelfFaceTeacher(1402)
        mem = SelfFaceMemory(max_views=8)
        for i in range(8):
            frame, bbox = teacher.render(2001, 2100+i)
            mem.enroll("me", frame, bbox)
        stranger, bbox = teacher.render(9001, 2500)
        row = mem.mark_not_me(stranger, bbox)
        self.assertLess(row["score_after"], row["score_before"])
        self.assertEqual(mem.summary()["negative_observations"], 1)

    def test_face_benchmark_keeps_no_raw_frames(self):
        rep = run_face_benchmark(seed=1403, enroll_views=6, test_views=10, negative_examples=2)
        self.assertTrue(rep.bounded_memory_ok)
        self.assertEqual(rep.raw_frames_retained, 0)
        self.assertGreaterEqual(rep.self_acceptance, 0.0)
        self.assertLessEqual(rep.self_acceptance, 1.0)

    def test_program_construction_memory_is_bounded(self):
        mem = ProgramConstructionMemory(max_patterns=9)
        self.assertEqual(mem.summary()["max_patterns"], 9)
        self.assertEqual(mem.summary()["patterns"], 0)

    def test_language_first_training_and_read_only_test(self):
        s = CognitiveSessionV14(seed=1404)
        before = s.language.learner.episode_count
        train = s.language_first_train(80)
        self.assertGreater(train["episodes_after"], before)
        before_test = s.language.learner.episode_count
        rep = s.test_rich_language(20)
        self.assertTrue(rep["memory_frozen"])
        self.assertEqual(s.language.learner.episode_count, before_test)
        self.assertIn("exact", rep)
        self.assertIn("program_patterns", rep)

    def test_v14_language_teacher_has_unseen_surface_constructions(self):
        teacher = RichSemanticTeacherV14(1405)
        train = teacher.v14_simple("QUERY", held_out=False)
        test = teacher.v14_simple("QUERY", held_out=True)
        self.assertTrue(train.utterance)
        self.assertTrue(test.utterance)
        self.assertNotEqual(train.utterance, test.utterance)
        self.assertEqual(train.program.intent(), "QUERY")
        self.assertEqual(test.program.intent(), "QUERY")

    def test_v14_checkpoint_persists_face_and_language_memory(self):
        teacher = SyntheticSelfFaceTeacher(1406)
        s = CognitiveSessionV14(seed=1406)
        s.language_first_train(25)
        frame, bbox = teacher.render(3001, 3101)
        s.enroll_self_face("me", frame, bbox)
        with tempfile.TemporaryDirectory() as td:
            s.save(td)
            restored = CognitiveSessionV14.load_checkpoint(td, seed=1406)
        self.assertEqual(restored.self_face.enrolled_name, "me")
        self.assertEqual(restored.language.learner.episode_count, s.language.learner.episode_count)
        self.assertEqual(restored.language_budget_ratio, .80)

    def test_v14_ui_and_release_surface(self):
        ui = Path("apcn_v14/ui.py").read_text(encoding="utf-8")
        launcher = Path("run_desktop_v0_14.py").read_text(encoding="utf-8")
        self.assertIn("Language First", ui)
        self.assertIn("Self Face Camera", ui)
        self.assertIn("Start Camera", ui)
        self.assertIn("Verify: Is This Me?", ui)
        self.assertIn("no neural face embedding encoder", ui)
        self.assertIn("APCNV14Window", launcher)


if __name__ == "__main__":
    unittest.main()
