#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from apcn_v07.trainer import train


def main() -> None:
    p = argparse.ArgumentParser(description="APCN V0.7 procedural grounded-concept trainer")
    p.add_argument("--episodes", type=int, default=2400, help="number of generated grounded experiences")
    p.add_argument("--eval-samples", type=int, default=400)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="outputs/v0_7")
    args = p.parse_args()

    learner, report = train(
        episodes=max(1, args.episodes),
        seed=args.seed,
        eval_samples=max(20, args.eval_samples),
        output_dir=args.output,
    )

    print("\n=== HELD-OUT GENERALIZATION ===")
    print(f"Color accuracy : {report.color_accuracy:.3f}")
    print(f"Shape accuracy : {report.shape_accuracy:.3f}")
    print(f"Joint accuracy : {report.joint_accuracy:.3f}")

    print("\n=== WHAT THE WORDS SELECTED ===")
    for word in ("yellow", "red", "circle", "square", "ellipse", "this", "is"):
        if word not in learner.token_stats:
            continue
        q = learner.concept_quality(word)
        mass = learner.diagnostic_group_mass(word)
        print(f"{word:>8s} quality={q:.3f}  diagnostic_signal_mass={mass}")

    out = Path(args.output)
    print(f"\nSaved compact memory: {out / 'concept_memory_v0_7.json'}")
    print(f"Saved report        : {out / 'training_report_v0_7.json'}")
    print("No individual training images were stored in long-term concept memory.")


if __name__ == "__main__":
    main()
