import tempfile
import unittest
from pathlib import Path

from apcn_v15.benchmark import run_conversation_benchmark
from apcn_v15.session import CognitiveSessionV15


class TestV015(unittest.TestCase):
    def _session(self):
        s = CognitiveSessionV15(seed=15001)
        for name in ("distance", "time", "velocity change", "mass", "volume"):
            s.concepts.add_primitive(name, grounded=True)
        s.concepts.learn_definition("speed is distance divided by time")
        s.concepts.learn_definition("acceleration is velocity change divided by time")
        return s

    def test_multiturn_definition_dependencies_and_why(self):
        s = self._session()
        r1 = s.talk("what is acceleration?")
        self.assertEqual(r1.act, "ANSWER_DEFINITION")
        self.assertIn("velocity change", r1.text.lower())
        r2 = s.talk("what does it depend on?")
        self.assertEqual(r2.act, "ANSWER_DEPENDENCIES")
        self.assertIn("time", r2.text.lower())
        r3 = s.talk("why?")
        self.assertEqual(r3.act, "ANSWER_PROVENANCE")
        self.assertIn("acceleration", r3.text.lower())

    def test_user_can_teach_alias_and_use_it_immediately(self):
        s = self._session()
        taught = s.talk("fluxion means acceleration")
        self.assertTrue(taught.learned)
        self.assertEqual(taught.act, "LEARN_ALIAS")
        answer = s.talk("what is fluxion?")
        self.assertEqual(answer.act, "ANSWER_DEFINITION")
        self.assertIn("acceleration", answer.text.lower())

    def test_user_can_teach_simple_fact(self):
        s = self._session()
        taught = s.talk("remember that orbix is a sensor")
        self.assertEqual(taught.act, "LEARN_FACT")
        answer = s.talk("what is orbix?")
        self.assertEqual(answer.act, "ANSWER_FACT")
        self.assertIn("sensor", answer.text.lower())
        yes = s.talk("is orbix a sensor?")
        self.assertIn("yes", yes.text.lower())

    def test_unknown_language_is_explicit_not_hallucinated(self):
        s = self._session()
        row = s.talk("please frobnicate the ontology sideways")
        self.assertEqual(row.act, "CLARIFY")
        self.assertLess(row.confidence, .5)
        self.assertIn("do not yet know", row.text.lower())

    def test_language_only_training_adds_no_visual_experiences(self):
        s = self._session()
        before = s.visual.learner.episode_count
        row = s.language_only_train(40)
        self.assertEqual(row["visual_experiences_added"], 0)
        self.assertEqual(s.visual.learner.episode_count, before)
        self.assertGreater(row["experiences_added"], 0)

    def test_conversational_memories_persist_without_raw_chat(self):
        s = self._session()
        s.talk("fluxion means acceleration")
        s.talk("remember that orbix is a sensor")
        s.talk("hello")
        with tempfile.TemporaryDirectory() as td:
            s.save(td)
            restored = CognitiveSessionV15.load_checkpoint(td, seed=15001)
            self.assertEqual(restored.lexicon_v15.resolve("fluxion")[0], "acceleration")
            self.assertEqual(restored.facts_v15.first_is_a("orbix").object, "sensor")
            state = Path(td, "session_v0_15.json").read_text(encoding="utf-8")
            self.assertNotIn("fluxion means acceleration", state)
            self.assertNotIn("remember that orbix is a sensor", state)
            self.assertFalse(restored.conversation_memory_audit()["raw_chat_transcript_persisted"])

    def test_conversation_benchmark(self):
        rep = run_conversation_benchmark(seed=15002)
        self.assertGreaterEqual(rep.act_accuracy, .85)
        self.assertGreaterEqual(rep.required_content_accuracy, .80)
        self.assertEqual(rep.unknown_honesty, 1.0)
        self.assertEqual(rep.learned_memory_accuracy, 1.0)
        self.assertGreaterEqual(rep.followup_accuracy, .80)
        self.assertEqual(rep.visual_experiences_changed, 0)

    def test_v15_ui_surface_is_language_only(self):
        ui = Path("apcn_v15/ui.py").read_text(encoding="utf-8")
        launcher = Path("run_desktop_v0_15.py").read_text(encoding="utf-8")
        self.assertIn("Conversation", ui)
        self.assertIn("Train Language Only", ui)
        self.assertIn("visual training budget = 0%", ui)
        self.assertIn("Self Face Camera", ui)  # removal of inherited tab is explicit
        self.assertIn("APCNV15Window", launcher)


if __name__ == "__main__":
    unittest.main()
