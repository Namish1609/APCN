import tempfile
import unittest
from pathlib import Path

from apcn_v10.language import AdaptiveLanguageSession, run_generated_language_test
from apcn_v10.definitions import ConceptStore, DefinitionCurriculum, DefinitionParseError


class TestV010(unittest.TestCase):
    def test_automatic_curriculum_reaches_all_skills(self):
        s = AdaptiveLanguageSession(seed=10)
        for _ in range(2200):
            s.step()
        for skill in s.SKILLS:
            self.assertGreater(s.skills[skill].attempts, 0, skill)
        self.assertGreater(s.learner.episode_count, 2200)

    def test_generated_language_test_is_read_only(self):
        s = AdaptiveLanguageSession(seed=11)
        for _ in range(1600):
            s.step()
        before = s.learner.episode_count
        rep = run_generated_language_test(s.learner, samples=180, seed=222)
        self.assertEqual(before, s.learner.episode_count)
        self.assertEqual(rep.learner_episode_count_before, rep.learner_episode_count_after)

    def test_language_semantics_improve(self):
        s = AdaptiveLanguageSession(seed=12)
        for _ in range(2800):
            s.step()
        rep = run_generated_language_test(s.learner, samples=360, seed=333)
        self.assertGreater(rep.relation_accuracy, 0.80)
        self.assertGreater(rep.intent_accuracy, 0.70)
        self.assertGreater(rep.operator_accuracy, 0.65)
        self.assertGreater(rep.exact_accuracy, 0.48)

    def test_arbitrary_definition_is_compositional(self):
        c = ConceptStore()
        c.add_primitive("dax")
        c.add_primitive("mip")
        rec = c.learn_definition("zorp is dax divided by mip")
        self.assertEqual(rec.definition.op, "DIV")
        self.assertEqual(c.evaluate("zorp", {"dax": 12, "mip": 3}), 4.0)
        self.assertTrue(c.understanding("zorp")["complete"])

    def test_definition_chain_executes(self):
        c = ConceptStore()
        for x in ("mass", "velocity change", "time"):
            c.add_primitive(x)
        c.learn_definition("acceleration is velocity change divided by time")
        c.learn_definition("force is the product of mass and acceleration")
        value = c.evaluate("force", {"mass": 2, "velocity change": 20, "time": 4})
        self.assertAlmostEqual(value, 10.0)
        self.assertEqual(c.dependency_depth("force"), 2)

    def test_grounding_audit_reports_unknown_dependency(self):
        c = ConceptStore()
        c.add_primitive("mass")
        c.learn_definition("mystery momentum is the product of mass and ghost velocity")
        audit = c.understanding("mystery momentum")
        self.assertFalse(audit["complete"])
        self.assertIn("ghost velocity", audit["unresolved"])

    def test_definition_cycle_rejected(self):
        c = ConceptStore()
        c.add_primitive("seed")
        c.learn_definition("alpha means seed")
        c.learn_definition("beta means alpha")
        with self.assertRaises(DefinitionParseError):
            c.learn_definition("alpha means beta")

    def test_concept_store_persistence(self):
        c = ConceptStore()
        c.add_primitive("distance")
        c.add_primitive("time")
        c.learn_definition("speed is distance divided by time")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "memory.json"
            c.save(p)
            d = ConceptStore.load(p)
            self.assertEqual(d.evaluate("speed", {"distance": 100, "time": 20}), 5.0)

    def test_science_bootstrap_definitions(self):
        curriculum = DefinitionCurriculum()
        curriculum.train_all_once()
        store = curriculum.store
        self.assertTrue(store.understanding("density")["complete"])
        self.assertTrue(store.understanding("force")["complete"])
        self.assertAlmostEqual(store.evaluate("density", {"mass": 10, "volume": 2}), 5.0)
        self.assertAlmostEqual(store.evaluate("force", {"mass": 2, "velocity change": 20, "time": 4}), 10.0)


if __name__ == "__main__":
    unittest.main()
