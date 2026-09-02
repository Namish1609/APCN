#!/usr/bin/env python3
from __future__ import annotations

import argparse

from apcn_v07.generator import ProceduralTeacher
from apcn_v07.learner import GroundedConceptLearner
from apcn_v07.trainer import evaluate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("memory", nargs="?", default="outputs/v0_7/concept_memory_v0_7.json")
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--difficulty", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=991)
    args = p.parse_args()

    learner = GroundedConceptLearner.load(args.memory)
    teacher = ProceduralTeacher(seed=args.seed)
    report = evaluate(learner, teacher, samples=args.samples, difficulty=args.difficulty)
    print(f"memory episodes : {learner.episode_count}")
    print(f"color accuracy  : {report.color_accuracy:.3f}")
    print(f"shape accuracy  : {report.shape_accuracy:.3f}")
    print(f"joint accuracy  : {report.joint_accuracy:.3f}")

    print("\nOpen vocabulary examples:")
    for _ in range(8):
        ep = teacher.generate(difficulty=args.difficulty)
        grounded = learner.ground_image(ep.image, ep.attention_mask, top_k=6)
        truth = f"{ep.teacher_metadata['color']} {ep.teacher_metadata['shape']}"
        pred = ", ".join(f"{g.token}:{g.score:.2f}" for g in grounded)
        print(f"  truth={truth:<18} grounded=[{pred}]")


if __name__ == "__main__":
    main()
