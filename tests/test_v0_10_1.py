import unittest

from apcn_v10.definitions import ConceptStore, DefinitionCurriculum
from apcn_v10.query import KnowledgeQueryEngine
from apcn_v10.language_v101 import AdaptiveLanguageSessionV101
from apcn_v10.language_session import run_generated_language_test


class TestV0101(unittest.TestCase):
    def test_definition_question_answering(self):
        store=ConceptStore(); curriculum=DefinitionCurriculum(store); curriculum.train_all_once()
        q=KnowledgeQueryEngine(store)
        ans=q.ask("what is acceleration?")
        self.assertEqual(ans.concept,"acceleration")
        self.assertIn("velocity change",ans.answer)
        self.assertIn("time",ans.answer)

    def test_executable_question(self):
        store=ConceptStore(); DefinitionCurriculum(store).train_all_once(); q=KnowledgeQueryEngine(store)
        ans=q.ask("calculate acceleration if velocity change = 20 and time = 4")
        self.assertIn("5",ans.answer)

    def test_unknown_question_is_explicit(self):
        q=KnowledgeQueryEngine(ConceptStore())
        self.assertIn("do not currently know",q.ask("what is unobtainium?").answer.lower())

    def test_long_intent_constructions_generalize(self):
        s=AdaptiveLanguageSessionV101(seed=101)
        while s.learner.episode_count < 3200:
            s.step()
        rep=run_generated_language_test(s.learner,samples=300,seed=10101)
        self.assertGreaterEqual(rep.intent_accuracy,0.88)
        self.assertGreaterEqual(rep.relation_accuracy,0.95)
        self.assertEqual(rep.learner_episode_count_before,rep.learner_episode_count_after)


if __name__ == "__main__":
    unittest.main()
