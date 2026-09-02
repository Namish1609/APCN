#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np

from apcn_v07.generator import GroundedEpisode
from apcn_v07.learner import GroundedConceptLearner


def parse_bbox(text: str) -> tuple[int, int, int, int]:
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be x,y,w,h")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("bbox width/height must be positive")
    return x, y, w, h


def main() -> None:
    p = argparse.ArgumentParser(description="Teach APCN V0.7 from a real image + joint attention")
    p.add_argument("--image", required=True)
    p.add_argument("--utterance", required=True)
    p.add_argument("--memory", default="outputs/v0_7/concept_memory_v0_7.json")
    focus = p.add_mutually_exclusive_group(required=True)
    focus.add_argument("--mask", help="grayscale/PNG mask; non-zero pixels are the pointed object")
    focus.add_argument("--bbox", type=parse_bbox, help="focus rectangle x,y,w,h")
    args = p.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"could not read image: {args.image}")

    if args.mask:
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise SystemExit(f"could not read mask: {args.mask}")
        if mask.shape[:2] != image.shape[:2]:
            raise SystemExit("mask size must match image size")
    else:
        x, y, w, h = args.bbox
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(image.shape[1], x + w), min(image.shape[0], y + h)
        if x1 <= x0 or y1 <= y0:
            raise SystemExit("bbox does not intersect image")
        mask[y0:y1, x0:x1] = 255

    memory = Path(args.memory)
    if memory.exists():
        learner = GroundedConceptLearner.load(memory)
    else:
        learner = GroundedConceptLearner()

    ep = GroundedEpisode(
        image=image,
        attention_mask=mask,
        utterance=args.utterance,
        teacher_metadata={"source": "real_image_user_joint_attention"},
    )
    learner.train_episode(ep)
    memory.parent.mkdir(parents=True, exist_ok=True)
    learner.save(memory)

    print(f"learned episode #{learner.episode_count}: {args.utterance!r}")
    print(f"memory saved to: {memory}")
    print("token qualities:")
    for token in learner.tokenize(args.utterance):
        print(f"  {token:>14s}: {learner.concept_quality(token):.4f}")


if __name__ == "__main__":
    main()
