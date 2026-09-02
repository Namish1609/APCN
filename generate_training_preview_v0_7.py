#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np

from apcn_v07.generator import ProceduralTeacher


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/v0_7/training_preview.jpg")
    p.add_argument("--seed", type=int, default=22)
    args = p.parse_args()

    teacher = ProceduralTeacher(seed=args.seed)
    tiles = []
    for i in range(20):
        ep = teacher.generate(difficulty=min(0.95, 0.15 + i * 0.04))
        img = ep.image.copy()
        cv2.putText(img, ep.utterance[:25], (4, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(img)
    rows = [np.hstack(tiles[i:i+5]) for i in range(0, 20, 5)]
    montage = np.vstack(rows)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), montage)
    print(path)


if __name__ == "__main__":
    main()
