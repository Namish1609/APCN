from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List

from .session import CognitiveSessionV19


@dataclass
class V19BenchmarkReport:
    basic_addition_accuracy: float
    basic_subtraction_accuracy: float
    english_recombination_accuracy: float
    reverse_subtraction_accuracy: float
    word_number_accuracy: float
    invented_operator_one_shot_accuracy: float
    invented_subtraction_one_shot_accuracy: float
    unknown_operator_honesty: float
    persistence_accuracy: float
    v018_fallback_accuracy: float
    no_backprop_contract: float
    examples: List[Dict[str, object]]
    benchmark_role: str = "finite_language_math_gate_not_general_llm_claim"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _answer_value(session: CognitiveSessionV19, text: str):
    row = session.talk(text)
    if row.act != "ANSWER_ARITHMETIC":
        return row, None
    try:
        value = float(row.text.rstrip(".").split("=")[-1].strip())
    except Exception:
        value = None
    return row, value


def run_language_math_benchmark(seed: int = 19019) -> V19BenchmarkReport:
    s = CognitiveSessionV19(seed=seed, bootstrap_english=True)
    examples: List[Dict[str, object]] = []

    add_cases = {
        "what is 37 plus 58?": 95.0,
        "calculate 1000 plus 2345": 3345.0,
        "please add 41 and 19": 60.0,
        "17 + 26": 43.0,
    }
    sub_cases = {
        "what is 91 minus 47?": 44.0,
        "calculate 80 minus 13": 67.0,
        "5 minus 12": -7.0,
        "30 - 19": 11.0,
    }
    recombination_cases = {
        "please calculate 21 plus 8": 29.0,
        "how much is 20 minus 7": 13.0,
        "could you calculate 44 plus 9": 53.0,
        "what is 50 minus 17": 33.0,
    }
    reverse_cases = {
        "subtract 7 from 30": 23.0,
        "take 4 away from 19": 15.0,
    }
    word_cases = {
        "what is seven plus five": 12.0,
        "how much is twenty minus six": 14.0,
    }

    def score(cases):
        passed = 0
        for text, expected in cases.items():
            row, value = _answer_value(s, text)
            ok = value is not None and abs(value - expected) < 1e-9
            passed += int(ok)
            examples.append({"input": text, "output": row.text, "act": row.act, "expected": expected, "pass": ok})
        return passed / max(1, len(cases))

    add_acc = score(add_cases)
    sub_acc = score(sub_cases)
    recomb_acc = score(recombination_cases)
    reverse_acc = score(reverse_cases)
    word_acc = score(word_cases)

    before = s.talk("what is 11 dax 8?")
    taught = s.talk("remember that 2 dax 3 equals 5")
    after, dax_value = _answer_value(s, "what is 11 dax 8?")
    invented_add = float(
        before.act != "ANSWER_ARITHMETIC"
        and taught.learned
        and dax_value is not None
        and abs(dax_value - 19.0) < 1e-9
    )
    examples.extend([
        {"input": "what is 11 dax 8?", "output": before.text, "phase": "before teaching"},
        {"input": "remember that 2 dax 3 equals 5", "output": taught.text, "phase": "teaching"},
        {"input": "what is 11 dax 8?", "output": after.text, "phase": "after teaching"},
    ])

    taught_sub = s.talk("remember that 9 nerk 4 equals 5")
    nerk, nerk_value = _answer_value(s, "please calculate 20 nerk 7")
    invented_sub = float(taught_sub.learned and nerk_value is not None and abs(nerk_value - 13.0) < 1e-9)
    examples.extend([
        {"input": "remember that 9 nerk 4 equals 5", "output": taught_sub.text},
        {"input": "please calculate 20 nerk 7", "output": nerk.text},
    ])

    unknown = s.talk("what is 9 florp 2?")
    unknown_honesty = float(unknown.act != "ANSWER_ARITHMETIC")
    examples.append({"input": "what is 9 florp 2?", "output": unknown.text, "act": unknown.act})

    with TemporaryDirectory() as td:
        s.save(td)
        restored = CognitiveSessionV19.load_checkpoint(td, seed=seed)
        persisted, persisted_value = _answer_value(restored, "calculate 13 dax 7")
        persistence = float(persisted_value is not None and abs(persisted_value - 20.0) < 1e-9)
        examples.append({"input": "calculate 13 dax 7", "output": persisted.text, "phase": "after reload"})

    s2 = CognitiveSessionV19(seed=seed + 1, bootstrap_english=True)
    s2.talk("remember that milo is a cat")
    fallback = s2.talk("is milo a cat?")
    fallback_acc = float(fallback.act == "ANSWER_TRUE")

    audit = s.v19_memory_audit()["architecture_contract"]
    no_backprop = float(
        audit["backpropagation"] is False
        and audit["gradient_descent"] is False
        and audit["external_llm"] is False
        and audit["pretrained_language_model"] is False
    )

    return V19BenchmarkReport(
        basic_addition_accuracy=add_acc,
        basic_subtraction_accuracy=sub_acc,
        english_recombination_accuracy=recomb_acc,
        reverse_subtraction_accuracy=reverse_acc,
        word_number_accuracy=word_acc,
        invented_operator_one_shot_accuracy=invented_add,
        invented_subtraction_one_shot_accuracy=invented_sub,
        unknown_operator_honesty=unknown_honesty,
        persistence_accuracy=persistence,
        v018_fallback_accuracy=fallback_acc,
        no_backprop_contract=no_backprop,
        examples=examples,
    )
