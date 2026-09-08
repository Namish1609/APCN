import tempfile
import unittest
from pathlib import Path

from apcn_v17.benchmark import run_structured_benchmark
from apcn_v17.language import StructuredTeacherV17
from apcn_v17.semantics import SemanticClause
from apcn_v17.session import CognitiveSessionV17


class TestV017(unittest.TestCase):
    def _session(self):
        return CognitiveSessionV17(seed=17011)

    def test_operator_bootstrap_excludes_recombination(self):
        s = self._session()
        self.assertEqual(s.structured_bootstrap_v17["recombination_templates_used"], 0)
        retained = {rec.template for rec in s.operator_memory_v17.records.values()}
        teacher = StructuredTeacherV17()
        for templates in teacher.RECOMBINATION.values():
            for template in templates:
                normalized = " ".join(template.lower().replace("{", " { ").replace("}", " } ").split())
                self.assertFalse(any(rec.template == normalized for rec in s.operator_memory_v17.records.values()))
        self.assertGreaterEqual(len(retained), 15)

    def test_structured_operator_parse(self):
        s = self._session()
        examples = [
            ("not milo is active", SemanticClause.make("NOT", children=(SemanticClause.make("PROPERTY", subject="milo", property="active", tense="present"),))),
            ("lamp turns off because power fails", SemanticClause.make("CAUSE", children=(SemanticClause.make("EVENT", subject="power", event="fail", tense="present"), SemanticClause.make("EVENT", subject="lamp", event="turn off", tense="present")))),
            ("if battery is empty then device stops", SemanticClause.make("IF", children=(SemanticClause.make("PROPERTY", subject="battery", property="empty", tense="present"), SemanticClause.make("EVENT", subject="device", event="stop", tense="present")))),
            ("door opens before light turns on", SemanticClause.make("BEFORE", children=(SemanticClause.make("EVENT", subject="door", event="open", tense="present"), SemanticClause.make("EVENT", subject="light", event="turn on", tense="present")))),
            ("every cat is a creature", SemanticClause.make("FORALL_ISA", kind="cat", category="creature")),
            ("some sensor is active", SemanticClause.make("EXISTS_PROPERTY", kind="sensor", property="active")),
        ]
        for surface, expected in examples:
            parsed, conf, evidence = s.structured_language_v17.parse(surface)
            self.assertEqual(parsed, expected, surface)
            self.assertGreater(conf, .5)
            self.assertTrue(evidence)

    def test_nested_roundtrip(self):
        s = self._session()
        clause = SemanticClause.make(
            "IF",
            children=(
                SemanticClause.make("NOT", children=(SemanticClause.make("PROPERTY", subject="battery", property="charged", tense="present"),)),
                SemanticClause.make("EVENT", subject="device", event="stop", tense="present"),
            ),
        )
        rep = s.structured_language_v17.roundtrip(clause, 12)
        self.assertGreaterEqual(rep["count"], 2)
        self.assertEqual(rep["roundtrip_exact"], 1.0)

    def test_discourse_reference_resolves_semantically(self):
        s = self._session()
        first = s.talk("remember that milo is a cat")
        second = s.talk("remember that it is active")
        query = s.talk("is it true that it is active?")
        self.assertTrue(first.learned)
        self.assertTrue(second.learned)
        self.assertEqual(s.discourse_v17.focus, "milo")
        self.assertEqual(query.act, "ANSWER_TRUE")
        self.assertFalse(s.discourse_v17.summary()["raw_chat_transcript_persisted"])

    def test_universal_rule_inference(self):
        s = self._session()
        s.talk("remember that every cat is a creature")
        s.talk("remember that milo is a cat")
        row = s.talk("is it true that milo is a creature?")
        self.assertEqual(row.act, "ANSWER_TRUE")
        self.assertTrue(any("universal_inference" in x for x in row.trace))

    def test_condition_cause_time_and_negation(self):
        s = self._session()
        s.talk("remember that if battery is empty then device stops")
        self.assertIn("device stops", s.talk("what follows if battery is empty?").text.lower())
        s.talk("remember that lamp turns off because power fails")
        self.assertIn("power fails", s.talk("what causes lamp turns off?").text.lower())
        s.talk("remember that door opens before light turns on")
        self.assertIn("door opens", s.talk("what happens before light turns on?").text.lower())
        s.talk("remember that not sensor is active")
        self.assertEqual(s.talk("is it true that sensor is active?").act, "ANSWER_FALSE")

    def test_unknown_honesty_and_truth_firewall(self):
        s = self._session()
        before = len(s.semantic_memory_v17.records)
        row = s.talk("is it true that zorbin is active?")
        self.assertEqual(row.act, "ANSWER_UNKNOWN")
        self.assertIn("zorbin", row.text.lower())
        self.assertEqual(len(s.semantic_memory_v17.records), before)
        self.assertFalse(hasattr(s.structured_language_v17, "semantic_memory"))
        self.assertFalse(hasattr(s.structured_language_v17, "world"))
        self.assertFalse(hasattr(s.operator_memory_v17, "world"))
        self.assertFalse(s.v17_memory_audit()["architecture_contract"]["surface_generator_has_truth_memory_reference"])

    def test_persistence_keeps_semantics_not_raw_chat(self):
        s = self._session()
        s.talk("remember that lamp turns off because power fails")
        s.talk("remember that milo is a cat")
        s.talk("remember that it is active")
        with tempfile.TemporaryDirectory() as td:
            s.save(td)
            restored = CognitiveSessionV17.load_checkpoint(td, seed=17011)
            self.assertGreaterEqual(len(restored.semantic_memory_v17.records), 3)
            self.assertEqual(restored.discourse_v17.focus, "milo")
            state = Path(td, "session_v0_17.json").read_text(encoding="utf-8")
            self.assertNotIn("remember that lamp turns off because power fails", state.lower())
            self.assertNotIn("remember that it is active", state.lower())

    def test_visual_state_unchanged_by_structured_language(self):
        s = self._session()
        before = s.visual.learner.episode_count
        s.talk("remember that if battery is empty then device stops")
        s.talk("what follows if battery is empty?")
        clause = SemanticClause.make("CAUSE", children=(SemanticClause.make("EVENT", subject="power", event="fail", tense="present"), SemanticClause.make("EVENT", subject="lamp", event="turn off", tense="present")))
        s.generate_structured(clause, 10)
        self.assertEqual(s.visual.learner.episode_count, before)

    def test_release_benchmark_contract(self):
        rep = run_structured_benchmark(seed=17012)
        self.assertGreaterEqual(rep.operator_parse_accuracy, .95)
        self.assertGreaterEqual(rep.nested_roundtrip_exact, .95)
        self.assertGreaterEqual(rep.generation_min_variants, 2)
        self.assertEqual(rep.discourse_reference_accuracy, 1.0)
        self.assertEqual(rep.universal_inference_accuracy, 1.0)
        self.assertEqual(rep.conditional_reasoning_accuracy, 1.0)
        self.assertEqual(rep.causal_retrieval_accuracy, 1.0)
        self.assertEqual(rep.temporal_order_accuracy, 1.0)
        self.assertEqual(rep.negation_accuracy, 1.0)
        self.assertEqual(rep.unknown_honesty, 1.0)
        self.assertEqual(rep.truth_firewall, 1.0)
        self.assertFalse(rep.raw_chat_persisted)
        self.assertEqual(rep.visual_experiences_changed, 0)

    def test_ui_launcher_and_release_metadata(self):
        ui = Path("apcn_v17/ui.py").read_text(encoding="utf-8")
        launcher = Path("run_desktop_v0_17.py").read_text(encoding="utf-8")
        doc = Path("V0_17_STRUCTURED_DISCOURSE.md").read_text(encoding="utf-8")
        self.assertIn("Structured Semantics", ui)
        self.assertIn("APCNV17Window", launcher)
        self.assertEqual(CognitiveSessionV17.VERSION, "0.17.0")
        self.assertIn("APCN V0.17", doc)
        self.assertIn("Structured Discourse Semantics", doc)


if __name__ == "__main__":
    unittest.main()
