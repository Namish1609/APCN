from __future__ import annotations

import argparse
import json
from pathlib import Path

from apcn_v12.session import CognitiveSessionV12


def main() -> int:
    p = argparse.ArgumentParser(description="APCN V0.12 self-organizing perception experiment")
    p.add_argument("--from-v11", action="store_true", help="import compatible outputs/v0_11 knowledge")
    p.add_argument("--v11-dir", default="outputs/v0_11")
    p.add_argument("--representation-bootstrap", type=int, default=240)
    p.add_argument("--visual", type=int, default=1600)
    p.add_argument("--language", type=int, default=0)
    p.add_argument("--visual-test", type=int, default=500)
    p.add_argument("--language-test", type=int, default=600)
    p.add_argument("--difficulty", type=float, default=.86)
    p.add_argument("--consolidation-cycles", type=int, default=2)
    p.add_argument("--visual-consolidation", type=int, default=500)
    p.add_argument("--language-consolidation", type=int, default=500)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--output", default="outputs/v0_12")
    args = p.parse_args()

    out = Path(args.output)
    if (out / "session_v0_12.json").exists() and not args.from_v11:
        s = CognitiveSessionV12.load_checkpoint(out, seed=args.seed)
        source = "existing V0.12 checkpoint"
    elif args.from_v11:
        s = CognitiveSessionV12.from_v11_checkpoint(
            args.v11_dir, seed=args.seed,
            representation_bootstrap=args.representation_bootstrap,
        )
        source = "V0.11 compatible knowledge + fresh V0.12 visual representation"
    else:
        s = CognitiveSessionV12(args.seed)
        if args.representation_bootstrap > 0:
            s.v012_bootstrap_experiences = s.visual.learner.bootstrap_representation(
                s.visual.teacher, args.representation_bootstrap, difficulty=.72)
        source = "clean V0.12"

    print(f"source: {source}")
    print(json.dumps(s.visual.learner.representation_summary(), indent=2))

    if args.visual > 0:
        print(f"training V0.12 visual concepts: {args.visual}")
        s.train_visual(args.visual)
    if args.language > 0:
        print(f"training language: {args.language}")
        s.train_language(args.language)

    cycles = []
    for i in range(max(0, args.consolidation_cycles)):
        print(f"consolidation cycle {i+1}/{args.consolidation_cycles}")
        cycles.append(s.consolidation_cycle(
            visual_test=min(args.visual_test, 500),
            language_test=min(args.language_test, 600),
            visual_train=args.visual_consolidation,
            language_train=args.language_consolidation,
            difficulty=args.difficulty,
        ))

    vr = s.test_visual(args.visual_test, args.difficulty)
    lr = s.test_language(args.language_test)
    paths = s.save(out)
    report = {
        "visual": {
            "color": vr.color_accuracy,
            "shape": vr.shape_accuracy,
            "joint": vr.joint_accuracy,
        },
        "language": {
            "exact": lr.exact_accuracy,
            "intent": lr.intent_accuracy,
            "relation": lr.relation_accuracy,
            "operator": lr.operator_accuracy,
            "reference": lr.skill_accuracy.get("reference", 0.0),
        },
        "representation": s.visual.learner.representation_summary(),
        "coverage": s.visual_coverage(),
        "cycles": cycles,
        "saved": paths,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
