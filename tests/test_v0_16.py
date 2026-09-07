import tempfile
import unittest
from pathlib import Path

from apcn_v16.bidirectional import SemanticFrame
from apcn_v16.compositional import RECOMBINATION_SPLIT
from apcn_v16.benchmark import run_bidirectional_benchmark
from apcn_v16.session import CognitiveSessionV16


class TestV016(unittest.TestCase):
    def _session(self):
        s = CognitiveSessionV16(seed=16011)
        for name in ("distance", "time", "velocity change", "mass", "volume"):
            s.concepts.add_primitive(name, grounded=True)
        s.concepts.learn_definition("speed is distance divided by time")
        s.concepts.learn_definition("acceleration is velocity change divided by time")
        s.concepts.learn_definition("density is mass divided by volume")
        s._rebuild_v16_language()
        return s

    def test_bootstrap_uses_no_dev_or_recombination_examples(self):
        s = self._session()
        self.assertEqual(s.bidirectional_bootstrap_v16["dev_templates_used"], 0)
        retained = {rec.template for rec in s.bidirectional_v16.records.values()}
        for templates in s.bidirectional_teacher_v16.DEV.values():
            for template in templates:
                self.assertNotIn(template, retained)
        for templates in RECOMBINATION_SPLIT.values():
            for template in templates:
                self.assertNotIn(template, retained)

    def test_same_construction_memory_parses_and_generates(self):
        s = self._session()
        parsed, conf, _ = s.bidirectional_v16.parse("what is acceleration", allowed_ops={"ASK_DEFINITION"})
        self.assertEqual(parsed, SemanticFrame.make("ASK_DEFINITION", concept="acceleration"))
        self.assertGreater(conf, .5)
        frame = SemanticFrame.make("STATE_DEFINITION", concept="acceleration", definition="velocity change divided by time")
        rows = s.bidirectional_v16.generate(frame, 20)
        self.assertGreaterEqual(len(rows), 5)
        rep = s.bidirectional_v16.roundtrip(frame, 20)
        self.assertEqual(rep["roundtrip_exact"], 1.0)

    def test_compositional_parser_recombines_train_cues_in_new_syntax(self):
        s = self._session()
        examples = [
            ("explain what acceleration means", SemanticFrame.make("ASK_DEFINITION", concept="acceleration")),
            ("tell me which concepts speed depends on", SemanticFrame.make("ASK_DEPENDENCIES", concept="speed")),
            ("is density familiar to you", SemanticFrame.make("ASK_KNOWLEDGE", concept="density")),
            ("explain the difference between speed and density", SemanticFrame.make("COMPARE", left="speed", right="density")),
        ]
        for text, expected in examples:
            parsed, conf, evidence = s.language_v16.parse(text)
            self.assertEqual(parsed, expected, text)
            self.assertGreater(conf, .5)
            self.assertTrue(any(row.get("mode") == "compositional_cue" for row in evidence))

    def test_conversation_uses_semantic_reasoner_then_realizer(self):
        s = self._session()
        row = s.talk("what is acceleration")
        self.assertEqual(row.act, "ANSWER_DEFINITION")
        self.assertIn("velocity change", row.text.lower())
        self.assertIn("time", row.text.lower())
        self.assertTrue(any("v016:reason:STATE_DEFINITION" == x for x in row.trace))
        self.assertTrue(any("v016:realize:construction_memory_only" == x for x in row.trace))

    def test_unknown_is_not_filled_from_language_statistics(self):
        s = self._session()
        before = set(s.concepts.records)
        row = s.talk("what is glorpnax")
        self.assertEqual(row.act, "CLARIFY")
        self.assertIn("glorpnax", row.text.lower())
        self.assertNotIn("glorpnax", s.concepts.records)
        self.assertEqual(before, set(s.concepts.records))

    def test_explicit_alias_teaching_transfers_into_v16(self):
        s = self._session()
        taught = s.talk("fluxion means acceleration")
        self.assertTrue(taught.learned)
        answer = s.talk("what is fluxion")
        self.assertEqual(answer.act, "ANSWER_DEFINITION")
        self.assertIn("velocity change", answer.text.lower())
        self.assertTrue(any("v016:reason:STATE_DEFINITION" == x for x in answer.trace))

    def test_surface_generator_has_no_world_memory_reference(self):
        s = self._session()
        self.assertFalse(hasattr(s.bidirectional_v16, "concepts"))
        self.assertFalse(hasattr(s.bidirectional_v16, "world"))
        self.assertFalse(hasattr(s.language_v16.memory, "concepts"))
        self.assertFalse(s.v16_memory_audit()["architecture_contract"]["surface_generator_has_world_memory_reference"])

    def test_generator_does_not_invent_unpassed_domain_facts(self):
        s = self._session()
        frame = SemanticFrame.make("STATE_DEFINITION", concept="milo flux", definition="zorbin divided by quux")
        rows = s.bidirectional_v16.generate(frame, 20)
        self.assertTrue(rows)
        forbidden = ("acceleration", "speed", "density", "force", "momentum", "pressure")
        for row in rows:
            self.assertIn("milo flux", row.lower())
            self.assertIn("zorbin", row.lower())
            self.assertIn("quux", row.lower())
            self.assertFalse(any(x in row.lower() for x in forbidden))

    def test_bidirectional_memory_persists_without_raw_chat(self):
        s = self._session()
        custom = SemanticFrame.make("ASK_DEFINITION", concept="orbix")
        s.teach_bidirectional_construction("please unpack orbix for me", custom, weight=5)
        s.talk("fluxion means acceleration")
        with tempfile.TemporaryDirectory() as td:
            s.save(td)
            restored = CognitiveSessionV16.load_checkpoint(td, seed=16011)
            parsed, _, _ = restored.bidirectional_v16.parse("please unpack orbix for me", allowed_ops={"ASK_DEFINITION"})
            self.assertEqual(parsed, custom)
            self.assertEqual(restored.lexicon_v15.resolve("fluxion")[0], "acceleration")
            state = Path(td, "session_v0_16.json").read_text(encoding="utf-8")
            self.assertNotIn("fluxion means acceleration", state)

    def test_generation_does_not_change_visual_training(self):
        s = self._session()
        before = s.visual.learner.episode_count
        for _ in range(10):
            s.talk("what is acceleration")
        frame = SemanticFrame.make("STATE_DEFINITION", concept="acceleration", definition="velocity change divided by time")
        s.generate_from_semantics(frame, 12)
        self.assertEqual(s.visual.learner.episode_count, before)

    def test_benchmark_architecture_contract(self):
        rep = run_bidirectional_benchmark(seed=16012)
        self.assertGreaterEqual(rep.request_accuracy, .95)
        self.assertGreaterEqual(rep.generation_min_variants, 5)
        self.assertGreaterEqual(rep.roundtrip_exact, .95)
        self.assertEqual(rep.unknown_honesty, 1.0)
        self.assertEqual(rep.explicit_learning_transfer, 1.0)
        self.assertEqual(rep.content_firewall, 1.0)
        self.assertEqual(rep.visual_experiences_changed, 0)
        self.assertGreaterEqual(rep.recombination_accuracy, .75)
        self.assertEqual(rep.benchmark_role, "development_and_architecture_contract_not_blind_final")

    def test_v16_release_metadata(self):
        # Root VERSION/README belong to the latest release and are expected to
        # advance. Historical regressions validate the immutable V0.16 release
        # document, launcher, and class version instead of pinning root metadata.
        release_doc = Path("V0_16_BIDIRECTIONAL_LANGUAGE.md").read_text(encoding="utf-8")
        launcher = Path("run_desktop_v0_16.py").read_text(encoding="utf-8")
        self.assertIn("APCN V0.16", release_doc)
        self.assertIn("Bidirectional Semantic Language", release_doc)
        self.assertIn("truth", release_doc.lower())
        self.assertIn("APCNV16Window", launcher)
        self.assertEqual(CognitiveSessionV16.VERSION, "0.16.0")

    def test_v16_ui_and_launcher_exist(self):
        ui = Path("apcn_v16/ui.py").read_text(encoding="utf-8")
        launcher = Path("run_desktop_v0_16.py").read_text(encoding="utf-8")
        self.assertIn("Bidirectional Language", ui)
        self.assertIn("Parse + Paraphrase", ui)
        self.assertIn("Generate + Roundtrip Test", ui)
        self.assertIn("outputs/v0_16", ui)
        self.assertIn("APCNV16Window", launcher)


if __name__ == "__main__":
    unittest.main()
