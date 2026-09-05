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
        s.concepts.learn_definition("density is mass divided by volume")
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

    def test_learned_dialogue_handles_heldout_surface_forms(self):
        s = self._session()
        a = s.talk("how would you define acceleration")
        self.assertEqual(a.act, "ANSWER_DEFINITION")
        self.assertIn("velocity change", a.text.lower())
        b = s.talk("which ideas feed into acceleration")
        self.assertEqual(b.act, "ANSWER_DEPENDENCIES")
        self.assertIn("time", b.text.lower())
        c = s.talk("would you say you understand density")
        self.assertEqual(c.act, "ANSWER_KNOWLEDGE")
        d = s.talk("set speed beside density conceptually")
        self.assertEqual(d.act, "ANSWER_COMPARE")
        self.assertIn("speed", d.text.lower())
        self.assertIn("density", d.text.lower())

    def test_dialogue_classifier_generalizes_read_only(self):
        s = self._session()
        before = s.dialogue_learner_v15.observations
        rep = s.test_dialogue_generalization(240)
        self.assertTrue(rep["memory_frozen"])
        self.assertEqual(s.dialogue_learner_v15.observations, before)
        self.assertGreaterEqual(rep["accuracy"], .70)

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
        self.assertGreater(row["dialogue_steps"], 0)
        self.assertGreater(row["grounded_semantic_steps"], 0)

    def test_english_exposure_is_bounded_surface_memory_not_semantics(self):
        s = self._session()
        text = (
            "A small robot learns language from repeated examples. "
            "The robot asks questions when a phrase is unfamiliar. "
            "Repeated language gives the robot surface familiarity."
        )
        row = s.ingest_english_text(text)
        self.assertGreater(row["tokens_added"], 10)
        self.assertFalse(row["semantic_learning"])
        self.assertFalse(row["raw_text_retained"])
        self.assertGreater(s.english_exposure_v15.familiarity("robot"), 0)
        self.assertFalse(s.concepts.understanding("robot").get("known"))
        summary = s.english_exposure_v15.summary()
        self.assertEqual(summary["raw_documents_retained"], 0)
        self.assertEqual(summary["raw_sentences_retained"], 0)

    def test_conversational_memories_persist_without_raw_chat_or_corpus(self):
        s = self._session()
        s.talk("fluxion means acceleration")
        s.talk("remember that orbix is a sensor")
        s.talk("hello")
        corpus = "A quasar can appear bright because it releases enormous energy."
        s.ingest_english_text(corpus)
        before_dialogue = s.dialogue_learner_v15.observations
        with tempfile.TemporaryDirectory() as td:
            s.save(td)
            restored = CognitiveSessionV15.load_checkpoint(td, seed=15001)
            self.assertEqual(restored.lexicon_v15.resolve("fluxion")[0], "acceleration")
            self.assertEqual(restored.facts_v15.first_is_a("orbix").object, "sensor")
            self.assertGreater(restored.english_exposure_v15.familiarity("quasar"), 0)
            self.assertEqual(restored.dialogue_learner_v15.observations, before_dialogue)
            state = Path(td, "session_v0_15.json").read_text(encoding="utf-8")
            exposure = Path(td, "english_exposure_v0_15.json").read_text(encoding="utf-8")
            self.assertNotIn("fluxion means acceleration", state)
            self.assertNotIn("remember that orbix is a sensor", state)
            self.assertNotIn(corpus, exposure)
            self.assertFalse(restored.conversation_memory_audit()["raw_chat_transcript_persisted"])

    def test_conversation_benchmark(self):
        rep = run_conversation_benchmark(seed=15002)
        self.assertGreaterEqual(rep.act_accuracy, .85)
        self.assertGreaterEqual(rep.required_content_accuracy, .80)
        self.assertGreaterEqual(rep.heldout_dialogue_act_accuracy, .70)
        self.assertGreaterEqual(rep.heldout_interactive_accuracy, .70)
        self.assertEqual(rep.unknown_honesty, 1.0)
        self.assertEqual(rep.learned_memory_accuracy, 1.0)
        self.assertGreaterEqual(rep.followup_accuracy, .80)
        self.assertEqual(rep.visual_experiences_changed, 0)

    def test_v15_ui_surface_is_language_only(self):
        ui = Path("apcn_v15/ui.py").read_text(encoding="utf-8")
        launcher = Path("run_desktop_v0_15.py").read_text(encoding="utf-8")
        corpus_ui = Path("apcn_v15/corpus_ui.py").read_text(encoding="utf-8")
        self.assertIn("Conversation", ui)
        self.assertIn("Train Language Only", ui)
        self.assertIn("visual training budget = 0%", ui)
        self.assertIn("Self Face Camera", ui)  # removal of inherited tab is explicit
        self.assertIn("APCNV15Window", launcher)
        self.assertIn("install_english_exposure_panel", launcher)
        self.assertIn("English Exposure", corpus_ui)


if __name__ == "__main__":
    unittest.main()
