import tempfile
import unittest
from pathlib import Path

from apcn_v10.definitions import ConceptStore
from apcn_v18.benchmark import run_natural_conversation_benchmark
from apcn_v18.realizer import AnswerPlanV18
from apcn_v18.session import CognitiveSessionV18


class TestV018(unittest.TestCase):
    def _session(self):
        s = CognitiveSessionV18(seed=18011)
        for name in ("distance", "time", "velocity change"):
            s.concepts.add_primitive(name, grounded=True)
        s.concepts.learn_definition("acceleration is velocity change divided by time")
        s.v18_binding_repairs = s._repair_knowledge_bindings()
        s._rebuild_v18_language()
        return s

    def test_canonical_concept_binding_repairs_detached_definition_store(self):
        s = CognitiveSessionV18(seed=18011)
        detached = ConceptStore()
        for name in ("velocity change", "time"):
            detached.add_primitive(name, grounded=True)
        detached.learn_definition("acceleration is velocity change divided by time")
        s.definitions.store = detached
        row = s._repair_knowledge_bindings()
        self.assertTrue(row["detached_definition_store_found"])
        self.assertIn("acceleration", s.concepts.records)
        self.assertIs(s.definitions.store, s.concepts)
        self.assertIs(s.query.store, s.concepts)

    def test_alias_is_available_on_first_query(self):
        s = self._session()
        taught = s.talk("fluxion means acceleration")
        self.assertTrue(taught.learned)
        first = s.talk("what is fluxion?")
        self.assertEqual(first.act, "ANSWER_DEFINITION")
        self.assertIn("velocity change", first.text.lower())
        self.assertIn("time", first.text.lower())

    def test_unknown_near_alias_offers_clarification_not_silent_binding(self):
        s = self._session()
        s.talk("fluxion means acceleration")
        row = s.talk("I asked what is Fluxation")
        self.assertEqual(row.act, "CLARIFY")
        self.assertIn("did you mean", row.text.lower())
        self.assertIn("fluxion", row.text.lower())
        self.assertNotIn("fluxation", s.lexicon_v15.aliases)

    def test_ordinary_property_and_isa_questions_use_structured_truth(self):
        s = self._session()
        s.talk("remember that milo is a cat")
        s.talk("remember that milo is active")
        prop = s.talk("is milo active?")
        isa = s.talk("is milo a cat?")
        self.assertEqual(prop.act, "ANSWER_TRUE")
        self.assertEqual(isa.act, "ANSWER_TRUE")
        self.assertTrue(any("v018:truth" in x for x in prop.trace))
        self.assertTrue(any("v018:truth" in x for x in isa.trace))

    def test_property_is_not_mirrored_as_legacy_category(self):
        s = self._session()
        s.talk("remember that milo is a cat")
        s.talk("remember that milo is active")
        rows = s.facts_v15.about("milo")
        categories = {r.object for r in rows if r.relation == "is_a"}
        self.assertIn("cat", categories)
        self.assertNotIn("active", categories)

    def test_about_aggregates_structured_entity_facts(self):
        s = self._session()
        s.talk("remember that milo is a cat")
        s.talk("remember that milo is active")
        row = s.talk("what do you know about milo?")
        self.assertEqual(row.act, "ANSWER_ABOUT")
        self.assertIn("cat", row.text.lower())
        self.assertIn("active", row.text.lower())

    def test_ordinary_isa_question_uses_universal_inference_and_why_uses_proof(self):
        s = self._session()
        s.talk("remember that milo is a cat")
        s.talk("remember that every cat is a creature")
        row = s.talk("is milo a creature?")
        self.assertEqual(row.act, "ANSWER_TRUE")
        self.assertTrue(any("inference" in x for x in row.trace))
        why = s.talk("why?")
        self.assertIn("cat", why.text.lower())
        self.assertIn("creature", why.text.lower())
        self.assertTrue(any("last_proof" in x for x in why.trace))

    def test_common_causative_surface_is_structural_cause(self):
        s = self._session()
        teach = s.talk("remember that power fails causes lamp turns off")
        self.assertTrue(teach.learned)
        row = s.talk("what causes lamp turns off?")
        self.assertEqual(row.act, "ANSWER_CAUSE")
        self.assertIn("power fails", row.text.lower())

    def test_unknown_truth_stays_unknown(self):
        s = self._session()
        before = len(s.semantic_memory_v17.records)
        row = s.talk("is zorbin a creature?")
        self.assertEqual(row.act, "ANSWER_UNKNOWN")
        self.assertIn("zorbin", row.text.lower())
        self.assertEqual(len(s.semantic_memory_v17.records), before)

    def test_natural_realizer_has_multiple_variants_and_no_truth_access(self):
        s = self._session()
        plan = AnswerPlanV18.make("ANSWER_TRUE_DERIVED", statement="milo is a creature", reason="milo is a cat, and every cat is a creature")
        rows = s.generate_answer_variants(plan, 12)["surfaces"]
        self.assertGreaterEqual(len(rows), 3)
        self.assertFalse(hasattr(s.natural_realizer_v18, "semantic_memory"))
        self.assertFalse(hasattr(s.natural_realizer_v18, "concepts"))
        self.assertFalse(hasattr(s.natural_realizer_v18, "world"))
        self.assertFalse(s.v18_memory_audit()["architecture_contract"]["surface_realizer_has_truth_memory_reference"])

    def test_persistence_keeps_semantics_not_raw_chat(self):
        s = self._session()
        s.talk("remember that milo is active")
        s.talk("is milo active?")
        with tempfile.TemporaryDirectory() as td:
            s.save(td)
            restored = CognitiveSessionV18.load_checkpoint(td, seed=18011)
            row = restored.talk("is milo active?")
            self.assertEqual(row.act, "ANSWER_TRUE")
            state = Path(td, "session_v0_18.json").read_text(encoding="utf-8").lower()
            self.assertNotIn("remember that milo is active", state)
            self.assertNotIn("is milo active", state)

    def test_benchmark_transcript_regression_contract(self):
        rep = run_natural_conversation_benchmark(seed=18012)
        self.assertEqual(rep.immediate_alias_accuracy, 1.0)
        self.assertEqual(rep.lexical_repair_accuracy, 1.0)
        self.assertEqual(rep.ordinary_property_truth_accuracy, 1.0)
        self.assertEqual(rep.ordinary_isa_truth_accuracy, 1.0)
        self.assertEqual(rep.about_aggregation_accuracy, 1.0)
        self.assertEqual(rep.universal_inference_accuracy, 1.0)
        self.assertEqual(rep.proof_followup_accuracy, 1.0)
        self.assertEqual(rep.conditional_accuracy, 1.0)
        self.assertEqual(rep.causal_accuracy, 1.0)
        self.assertEqual(rep.temporal_accuracy, 1.0)
        self.assertEqual(rep.unknown_honesty, 1.0)
        self.assertEqual(rep.property_category_separation, 1.0)
        self.assertGreaterEqual(rep.natural_realization_min_variants, 3)
        self.assertEqual(rep.content_firewall, 1.0)
        self.assertEqual(rep.visual_experiences_changed, 0)
        self.assertIn("not_blind", rep.benchmark_role)

    def test_ui_launcher_and_release_metadata(self):
        ui = Path("apcn_v18/ui.py").read_text(encoding="utf-8")
        launcher = Path("run_desktop_v0_18.py").read_text(encoding="utf-8")
        version = Path("VERSION").read_text(encoding="utf-8")
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("Conversation Quality", ui)
        self.assertIn("APCNV18Window", launcher)
        self.assertIn("0.18.0", version)
        self.assertIn("V0.18", readme)
        self.assertIn("run_desktop_v0_18.py", readme)


if __name__ == "__main__":
    unittest.main()
