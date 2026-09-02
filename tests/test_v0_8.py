from __future__ import annotations
import tempfile
import unittest

from apcn_v08.session import TrainingSessionV08


class APCNV08Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = TrainingSessionV08(seed=18)
        for _ in range(1200):
            cls.session.step()

    def test_content_words_are_visually_grounded(self):
        self.assertEqual(self.session.learner.role_guess("yellow"), "visually_grounded")
        self.assertEqual(self.session.learner.role_guess("circle"), "visually_grounded")

    def test_function_words_are_not_visual_concepts(self):
        self.assertEqual(self.session.learner.role_guess("this"), "structural_or_abstract")
        self.assertLess(self.session.learner.concept_quality("this"), 0.08)

    def test_relevance_profiles_disentangle_factors(self):
        l = self.session.learner
        self.assertGreater(l.relevance_similarity("yellow", "red"), l.relevance_similarity("yellow", "circle") + 0.30)
        self.assertGreater(l.relevance_similarity("circle", "square"), l.relevance_similarity("circle", "yellow") + 0.30)

    def test_discover_families_has_color_like_component(self):
        families = self.session.learner.discover_families()
        member_sets = [set(f["members"]) for f in families]
        self.assertTrue(any({"yellow", "red", "green"}.issubset(m) for m in member_sets))

    def test_activation_trace_is_sparse_and_inspectable(self):
        r = self.session.generate_preview("yellow", "circle", 0.8)
        trace = self.session.learner.activation_trace(r.features, r.episode.utterance)
        self.assertGreater(len(trace["nodes"]), 4)
        self.assertGreater(len(trace["edges"]), 2)
        self.assertTrue(any(n["id"] == "word:yellow" for n in trace["nodes"]))
        self.assertTrue(any(str(n["id"]).startswith("feature:") for n in trace["nodes"]))

    def test_preview_does_not_train(self):
        before = self.session.learner.episode_count
        self.session.generate_preview("yellow", "circle", 0.5)
        self.assertEqual(before, self.session.learner.episode_count)

    def test_save_load_resume(self):
        with tempfile.TemporaryDirectory() as td:
            path = self.session.save(td)
            loaded = TrainingSessionV08.load(path, seed=18)
            self.assertEqual(loaded.learner.episode_count, self.session.learner.episode_count)
            before = loaded.learner.episode_count
            loaded.step()
            self.assertEqual(loaded.learner.episode_count, before + 1)
            self.assertEqual(loaded.curriculum.index, loaded.learner.episode_count)


if __name__ == "__main__":
    unittest.main()
