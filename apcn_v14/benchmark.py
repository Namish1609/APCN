from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import json

from apcn_v10.semantic import semantic_equal
from .language_teacher import RichSemanticTeacherV14
from .session import CognitiveSessionV14


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
    per_kind_exact: Dict[str, float]
    failure_kinds: Dict[str, int]

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


def run_v14_language_benchmark(seed: int = 14014, *, train_steps: int = 1800,
                                test_samples: int = 300) -> V14LanguageBenchmark:
    cognitive = CognitiveSessionV14(seed)
    session = cognitive.language
    cognitive.language_first_train(max(1, int(train_steps)))

    teacher = RichSemanticTeacherV14(seed + 701)
    before = session.learner.episode_count
    exact = intent = relation = operator = 0
    kinds = ("ASSERT", "QUERY", "GOAL", "GROUP", "NEGATE")
    kind_total = {k: 0 for k in kinds}
    kind_exact = {k: 0 for k in kinds}
    failure_kinds: Dict[str, int] = {}
    for i in range(max(1, int(test_samples))):
        ep = teacher.held_out_construction(i)
        kind = kinds[i % len(kinds)]
        pred = session.learner.parse(ep.utterance)
        ok = semantic_equal(pred, ep.program)
        exact += int(ok)
        kind_total[kind] += 1
        kind_exact[kind] += int(ok)
        intent_ok = pred is not None and pred.intent() == ep.program.intent()
        relation_ok = pred is not None and pred.relations() == ep.program.relations()
        operator_ok = pred is not None and pred.operators() == ep.program.operators()
        intent += int(intent_ok)
        relation += int(relation_ok)
        operator += int(operator_ok)
        if not ok:
            if pred is None:
                fk = "unresolved"
            elif not operator_ok:
                fk = "operator"
            elif not intent_ok:
                fk = "intent"
            elif not relation_ok:
                fk = "relation"
            else:
                fk = "arguments_or_structure"
            failure_kinds[fk] = failure_kinds.get(fk, 0) + 1
    after = session.learner.episode_count

    # Exactly ONE explicit human demonstration. V0.14 gives this event strong
    # local construction weight without repeating full lexical co-occurrence
    # learning, so an arbitrary function cue cannot become a phantom object word.
    base = teacher.v14_simple("GOAL", held_out=False, skill="user_paraphrase")
    words = base.utterance.split()
    custom = "zibble " + (" ".join(words[1:]) if len(words) > 1 else base.utterance)
    taught = cognitive.teach_language_paraphrase(custom, base.program)
    paraphrase_ok = 1.0 if taught["matched_after"] else 0.0

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
        per_kind_exact={k: kind_exact[k]/max(1, kind_total[k]) for k in kinds},
        failure_kinds=failure_kinds,
    )


def main() -> int:
    rep = run_v14_language_benchmark()
    print(json.dumps(rep.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
