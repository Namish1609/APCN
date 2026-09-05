from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import json

from apcn_v10.semantic import semantic_equal
from .language import AdaptiveLanguageSessionV14
from .language_teacher import RichSemanticTeacherV14


@dataclass
class V14LanguageBenchmark:
    train_steps: int
    test_samples: int
    exact_accuracy: float
    intent_accuracy: float
    relation_accuracy: float
    operator_accuracy: float
    user_paraphrase_after_teaching: float
    memory_frozen_during_test: bool
    program_patterns: int
    bounded_program_memory: bool

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


def run_v14_language_benchmark(seed: int = 14014, *, train_steps: int = 1800,
                                test_samples: int = 300) -> V14LanguageBenchmark:
    session = AdaptiveLanguageSessionV14(seed)
    for _ in range(max(1, int(train_steps))):
        session.step()

    teacher = RichSemanticTeacherV14(seed + 701)
    before = session.learner.episode_count
    exact = intent = relation = operator = 0
    for i in range(max(1, int(test_samples))):
        ep = teacher.held_out_construction(i)
        pred = session.learner.parse(ep.utterance)
        exact += int(semantic_equal(pred, ep.program))
        intent += int(pred is not None and pred.intent() == ep.program.intent())
        relation += int(pred is not None and pred.relations() == ep.program.relations())
        operator += int(pred is not None and pred.operators() == ep.program.operators())
    after = session.learner.episode_count

    # Demonstrate direct human language teaching without global retraining:
    # an arbitrary verb is taught as a paraphrase of a grounded GOAL program.
    base = teacher.v14_simple("GOAL", held_out=False, skill="user_paraphrase")
    words = base.utterance.split()
    # Keep grounded entity/relation words, replace the command construction cue.
    if len(words) > 1:
        custom = "zibble " + " ".join(words[1:])
    else:
        custom = "zibble " + base.utterance
    for _ in range(4):
        session.teach_user_paraphrase(custom, base.program)
    learned = session.learner.parse(custom)
    paraphrase_ok = 1.0 if semantic_equal(learned, base.program) else 0.0

    summary = session.learner.program_constructions.summary()
    n = max(1, int(test_samples))
    return V14LanguageBenchmark(
        train_steps=int(train_steps),
        test_samples=n,
        exact_accuracy=exact/n,
        intent_accuracy=intent/n,
        relation_accuracy=relation/n,
        operator_accuracy=operator/n,
        user_paraphrase_after_teaching=paraphrase_ok,
        memory_frozen_during_test=(before == after),
        program_patterns=int(summary["patterns"]),
        bounded_program_memory=int(summary["patterns"]) <= int(summary["max_patterns"]),
    )


def main() -> int:
    rep = run_v14_language_benchmark()
    print(json.dumps(rep.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
