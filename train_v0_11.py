from __future__ import annotations

import argparse
import json

from apcn_v11.session import CognitiveSessionV11


def main() -> int:
    p = argparse.ArgumentParser(description="APCN V0.11 unified-memory/consolidation experiment")
    p.add_argument("--visual", type=int, default=2000, help="visual training experiences")
    p.add_argument("--language", type=int, default=3200, help="language training experiences")
    p.add_argument("--visual-test", type=int, default=500)
    p.add_argument("--language-test", type=int, default=600)
    p.add_argument("--difficulty", type=float, default=.82)
    p.add_argument("--definitions", action="store_true")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--output", default="outputs/v0_11")
    args = p.parse_args()

    s = CognitiveSessionV11(args.seed)
    print(f"training visual: {args.visual}")
    s.train_visual(args.visual)
    print(f"training language: {args.language}")
    s.train_language(args.language)
    if args.definitions:
        s.learn_definition_curriculum()

    vr = s.test_visual(args.visual_test, args.difficulty)
    lr = s.test_language(args.language_test)
    graph = s.sync_graph()
    prescriptions = s.prescriptions(12)
    paths = s.save(args.output)

    report = {
        "visual": {"color": vr.color_accuracy, "shape": vr.shape_accuracy, "joint": vr.joint_accuracy},
        "language": {"exact": lr.exact_accuracy, "intent": lr.intent_accuracy, "relation": lr.relation_accuracy, "operator": lr.operator_accuracy},
        "graph": graph,
        "memory": s.memory_audit(),
        "next_learning_prescriptions": [x.__dict__ for x in prescriptions],
        "saved": paths,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
