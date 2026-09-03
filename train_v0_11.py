from __future__ import annotations

import argparse
import json

from apcn_v11.session import CognitiveSessionV11


def main() -> int:
    p = argparse.ArgumentParser(description="APCN V0.11 unified-memory/consolidation experiment")
    p.add_argument("--visual", type=int, default=2000, help="new visual training experiences")
    p.add_argument("--language", type=int, default=3200, help="new language training experiences")
    p.add_argument("--visual-test", type=int, default=500)
    p.add_argument("--language-test", type=int, default=600)
    p.add_argument("--difficulty", type=float, default=.82)
    p.add_argument("--definitions", action="store_true")
    p.add_argument("--consolidation-cycles", type=int, default=0)
    p.add_argument("--visual-memory", default=None, help="existing V0.8/V0.10 visual memory to migrate")
    p.add_argument("--language-memory", default=None, help="existing V0.10 language memory to migrate")
    p.add_argument("--concept-memory", default=None, help="existing V0.10 concept store to migrate")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--output", default="outputs/v0_11")
    args = p.parse_args()

    s = CognitiveSessionV11.from_memories(
        seed=args.seed,
        visual_memory=args.visual_memory,
        language_memory=args.language_memory,
        concept_memory=args.concept_memory,
    )
    print(f"starting memory: visual={s.visual.learner.episode_count}, language={s.language.learner.episode_count}")
    print(f"training new visual experiences: {args.visual}")
    s.train_visual(args.visual)
    print(f"training new language experiences: {args.language}")
    s.train_language(args.language)
    if args.definitions:
        s.learn_definition_curriculum()

    cycles = []
    for i in range(max(0, args.consolidation_cycles)):
        print(f"consolidation cycle {i+1}/{args.consolidation_cycles}")
        cycles.append(s.consolidation_cycle(
            visual_test=min(args.visual_test, 400),
            language_test=min(args.language_test, 480),
            visual_train=400,
            language_train=500,
            difficulty=args.difficulty,
        ))

    vr = s.test_visual(args.visual_test, args.difficulty)
    lr = s.test_language(args.language_test)
    graph = s.sync_graph()
    prescriptions = s.prescriptions(12)
    paths = s.save(args.output)

    report = {
        "visual": {"color": vr.color_accuracy, "shape": vr.shape_accuracy, "joint": vr.joint_accuracy},
        "language": {"exact": lr.exact_accuracy, "intent": lr.intent_accuracy, "relation": lr.relation_accuracy, "operator": lr.operator_accuracy},
        "consolidation_cycles": cycles,
        "graph": graph,
        "memory": s.memory_audit(),
        "next_learning_prescriptions": [x.__dict__ for x in prescriptions],
        "saved": paths,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
