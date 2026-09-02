from __future__ import annotations
import tempfile, unittest
from pathlib import Path

from apcn_v09.learner import SemanticLanguageLearner
from apcn_v09.semantic import semantic_equal
from apcn_v09.session import SemanticSessionV09
from apcn_v09.teacher import Lexicon, SemanticTeacher
from apcn_v09.testing import run_semantic_test


class V09Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = SemanticSessionV09(seed=91)
        cls.session.train(2600)

    def test_literal_relation_grounding(self):
        top = self.session.learner.top_cues("relation:R0", 5)
        cues = [x.cue for x in top]
        self.assertTrue(any("inside" in c for c in cues), cues)

    def test_intent_contrast(self):
        t = SemanticTeacher(seed=92)
        for intent in ("ASSERT", "QUERY", "GOAL"):
            ep = t.simple(intent=intent, relation="R1")
            pred = self.session.learner.parse(ep.utterance)
            self.assertIsNotNone(pred)
            self.assertEqual(pred.intent(), intent, (ep.utterance, pred.pretty()))

    def test_group_and_sequence(self):
        t = SemanticTeacher(seed=93)
        for ep in (t.group(), t.sequence(), t.negated()):
            pred = self.session.learner.parse(ep.utterance, ep.discourse_focus)
            self.assertTrue(semantic_equal(pred, ep.program), (ep.utterance, ep.program.pretty(), None if pred is None else pred.pretty()))

    def test_arbitrary_vocabulary(self):
        t = SemanticTeacher(seed=94, lexicon=Lexicon.scrambled())
        l = SemanticLanguageLearner()
        for i in range(2200):
            for ep in t.curriculum_episode(i):
                l.observe(ep)
        ep = t.simple(intent="GOAL", relation="R0")
        pred = l.parse(ep.utterance)
        self.assertIsNotNone(pred)
        self.assertEqual(pred.relations()[0], "R0")
        self.assertTrue(any(x.cue == "zorp" for x in l.top_cues("relation:R0", 8)))

    def test_read_only_benchmark(self):
        before = self.session.learner.episode_count
        rep = run_semantic_test(self.session.learner, 120, seed=951)
        self.assertEqual(before, self.session.learner.episode_count)
        self.assertEqual(rep.learner_episode_count_before, rep.learner_episode_count_after)

    def test_benchmark_threshold(self):
        rep = run_semantic_test(self.session.learner, 300, seed=961)
        self.assertGreaterEqual(rep.intent_accuracy, .80)
        self.assertGreaterEqual(rep.relation_accuracy, .85)
        self.assertGreaterEqual(rep.operator_accuracy, .80)
        self.assertGreaterEqual(rep.exact_accuracy, .60)

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            self.session.learner.save(p)
            loaded = SemanticLanguageLearner.load(p)
            self.assertEqual(loaded.episode_count, self.session.learner.episode_count)
            ep = SemanticTeacher(seed=99).simple(intent="QUERY")
            a = self.session.learner.parse(ep.utterance)
            b = loaded.parse(ep.utterance)
            self.assertEqual(None if a is None else a.canonical(), None if b is None else b.canonical())


if __name__ == "__main__":
    unittest.main()
