import tempfile
import unittest
from pathlib import Path

from apcn_v19.benchmark import run_language_math_benchmark
from apcn_v19.language_math import OnlineConstructionArithmeticV19
from apcn_v19.session import CognitiveSessionV19


class TestV019LanguageMath(unittest.TestCase):
    def test_bootstrap_generalizes_to_unseen_numbers(self):
        s = CognitiveSessionV19(seed=19001)
        row = s.talk("what is 37 plus 58?")
        self.assertEqual(row.act, "ANSWER_ARITHMETIC")
        self.assertIn("95", row.text)
        row = s.talk("what is 91 minus 47?")
        self.assertEqual(row.act, "ANSWER_ARITHMETIC")
        self.assertIn("44", row.text)

    def test_recombines_wrapper_and_operator_evidence(self):
        s = CognitiveSessionV19(seed=19002)
        row = s.talk("please calculate 21 plus 8")
        self.assertEqual(row.act, "ANSWER_ARITHMETIC")
        self.assertIn("29", row.text)
        row = s.talk("how much is 20 minus 7")
        self.assertEqual(row.act, "ANSWER_ARITHMETIC")
        self.assertIn("13", row.text)

    def test_reverse_subtraction_constructions(self):
        s = CognitiveSessionV19(seed=19003)
        row = s.talk("subtract 7 from 30")
        self.assertEqual(row.act, "ANSWER_ARITHMETIC")
        self.assertEqual(row.text, "30 - 7 = 23.")
        row = s.talk("take 4 away from 19")
        self.assertEqual(row.act, "ANSWER_ARITHMETIC")
        self.assertEqual(row.text, "19 - 4 = 15.")

    def test_word_numbers(self):
        s = CognitiveSessionV19(seed=19004)
        self.assertEqual(s.talk("what is seven plus five").text, "7 + 5 = 12.")
        self.assertEqual(s.talk("how much is twenty minus six").text, "20 - 6 = 14.")

    def test_one_shot_invented_addition_word(self):
        s = CognitiveSessionV19(seed=19005)
        before = s.talk("what is 11 dax 8?")
        self.assertNotEqual(before.act, "ANSWER_ARITHMETIC")
        taught = s.talk("remember that 2 dax 3 equals 5")
        self.assertTrue(taught.learned)
        row = s.talk("what is 11 dax 8?")
        self.assertEqual(row.act, "ANSWER_ARITHMETIC")
        self.assertEqual(row.text, "11 + 8 = 19.")

    def test_one_shot_invented_subtraction_word(self):
        s = CognitiveSessionV19(seed=19006)
        taught = s.talk("remember that 9 nerk 4 equals 5")
        self.assertTrue(taught.learned)
        row = s.talk("please calculate 20 nerk 7")
        self.assertEqual(row.act, "ANSWER_ARITHMETIC")
        self.assertEqual(row.text, "20 - 7 = 13.")

    def test_direct_operator_language_teaching(self):
        core = OnlineConstructionArithmeticV19(bootstrap=False)
        learned = core.teach_operator_phrase("zibble", "addition")
        self.assertTrue(learned["learned"])
        row = core.infer("12 zibble 5")
        self.assertTrue(row.understood)
        self.assertEqual(row.result, 17.0)

    def test_unknown_operator_stays_unknown(self):
        s = CognitiveSessionV19(seed=19007)
        row = s.talk("what is 9 florp 2?")
        self.assertNotEqual(row.act, "ANSWER_ARITHMETIC")

    def test_persistence_keeps_online_language_learning(self):
        s = CognitiveSessionV19(seed=19008)
        s.talk("remember that 2 dax 3 equals 5")
        with tempfile.TemporaryDirectory() as td:
            s.save(td)
            restored = CognitiveSessionV19.load_checkpoint(td, seed=19008)
            row = restored.talk("calculate 13 dax 7")
            self.assertEqual(row.act, "ANSWER_ARITHMETIC")
            self.assertEqual(row.text, "13 + 7 = 20.")
            self.assertTrue(Path(td, "language_math_v0_19.json").exists())

    def test_v018_semantic_conversation_remains_available(self):
        s = CognitiveSessionV19(seed=19009)
        s.talk("remember that milo is a cat")
        row = s.talk("is milo a cat?")
        self.assertEqual(row.act, "ANSWER_TRUE")

    def test_no_backprop_contract(self):
        s = CognitiveSessionV19(seed=19010)
        audit = s.v19_memory_audit()["architecture_contract"]
        self.assertFalse(audit["backpropagation"])
        self.assertFalse(audit["gradient_descent"])
        self.assertFalse(audit["external_llm"])
        self.assertFalse(audit["pretrained_language_model"])

    def test_finite_benchmark_gate(self):
        rep = run_language_math_benchmark(seed=19011)
        self.assertEqual(rep.basic_addition_accuracy, 1.0)
        self.assertEqual(rep.basic_subtraction_accuracy, 1.0)
        self.assertEqual(rep.english_recombination_accuracy, 1.0)
        self.assertEqual(rep.reverse_subtraction_accuracy, 1.0)
        self.assertEqual(rep.word_number_accuracy, 1.0)
        self.assertEqual(rep.invented_operator_one_shot_accuracy, 1.0)
        self.assertEqual(rep.invented_subtraction_one_shot_accuracy, 1.0)
        self.assertEqual(rep.unknown_operator_honesty, 1.0)
        self.assertEqual(rep.persistence_accuracy, 1.0)
        self.assertEqual(rep.v018_fallback_accuracy, 1.0)
        self.assertEqual(rep.no_backprop_contract, 1.0)
        self.assertIn("finite", rep.benchmark_role)


if __name__ == "__main__":
    unittest.main()
