from __future__ import annotations
import argparse
import json

from apcn_v10.session import CognitiveSessionV10


def main() -> int:
    p=argparse.ArgumentParser(description="Train APCN V0.10 automatic language and definitions")
    p.add_argument("--language-experiences",type=int,default=3000)
    p.add_argument("--test-samples",type=int,default=600)
    p.add_argument("--definitions",action="store_true")
    p.add_argument("--seed",type=int,default=10)
    args=p.parse_args()
    s=CognitiveSessionV10(args.seed)
    target=max(0,args.language_experiences)
    start=s.language.learner.episode_count
    while s.language.learner.episode_count-start < target:
        s.language.step()
        n=s.language.learner.episode_count-start
        if n and n % 250 == 0:
            print(f"language {n}/{target} ({100*n/max(1,target):.1f}%) skill={s.language.last_skill}")
    if args.definitions:
        s.definitions.train_all_once()
    rep=s.test_language(args.test_samples)
    print(json.dumps({
        "language_episodes": s.language.learner.episode_count,
        "competence": s.language.competence(),
        "generated_test": {
            "exact": rep.exact_accuracy,
            "intent": rep.intent_accuracy,
            "relation": rep.relation_accuracy,
            "operator": rep.operator_accuracy,
            "skill_accuracy": rep.skill_accuracy,
        },
        "definitions": s.concepts.definition_count,
    },indent=2))
    print(s.save())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
