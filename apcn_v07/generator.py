from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import math
import random
import cv2
import numpy as np


@dataclass
class GroundedEpisode:
    image: np.ndarray
    attention_mask: np.ndarray
    utterance: str
    # Metadata below belongs to the teacher/evaluator. Learner.train_episode()
    # deliberately ignores it.
    teacher_metadata: Dict[str, object]


class ProceduralTeacher:
    """
    Generates grounded English experiences automatically.

    The teacher knows the synthetic world's factors because it creates them.
    APCN is only given pixels, a joint-attention mask, and the utterance.
    """

    COLORS: Dict[str, Tuple[int, int, int]] = {
        # OpenCV BGR. Values are intentionally not exposed to the learner.
        "yellow": (30, 215, 235),
        "red": (45, 55, 220),
        "green": (65, 185, 75),
        "blue": (210, 105, 45),
        "purple": (180, 70, 170),
        "orange": (35, 135, 235),
    }
    SHAPES: Tuple[str, ...] = ("circle", "square", "ellipse", "triangle", "rectangle")

    TEMPLATES: Tuple[str, ...] = (
        "this is a {color} {shape}",
        "look at the {color} {shape}",
        "the {shape} is {color}",
        "that object is a {color} {shape}",
        "here is a {shape} that is {color}",
        "{color} {shape}",
        "notice this {color} {shape}",
    )

    def __init__(self, seed: int = 7, canvas_size: int = 160):
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed)
        self.canvas_size = int(canvas_size)

    @property
    def color_words(self) -> List[str]:
        return list(self.COLORS.keys())

    @property
    def shape_words(self) -> List[str]:
        return list(self.SHAPES)

    def _background(self, difficulty: float) -> np.ndarray:
        s = self.canvas_size
        base = self.rng.randint(205, 245)
        bg = np.full((s, s, 3), base, dtype=np.float64)
        if difficulty > 0:
            gx = np.linspace(-1.0, 1.0, s)[None, :, None]
            gy = np.linspace(-1.0, 1.0, s)[:, None, None]
            slope_x = self.rng.uniform(-18, 18) * difficulty
            slope_y = self.rng.uniform(-18, 18) * difficulty
            bg += gx * slope_x + gy * slope_y
            sigma = 2.0 + 8.0 * difficulty
            bg += self.nprng.normal(0.0, sigma, size=bg.shape)
        return np.clip(bg, 0, 255).astype(np.uint8)

    def _adjust_color(self, word: str, difficulty: float) -> Tuple[int, int, int]:
        base = np.asarray(self.COLORS[word], dtype=np.float64)
        brightness = self.rng.uniform(0.82 - 0.12 * difficulty, 1.10 + 0.08 * difficulty)
        jitter = self.nprng.normal(0, 4.0 + 7.0 * difficulty, size=3)
        out = base * brightness + jitter
        return tuple(int(x) for x in np.clip(out, 15, 245))

    def _draw_shape(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        shape: str,
        center: Tuple[int, int],
        radius: int,
        color: Tuple[int, int, int],
        rotation_deg: float,
    ) -> None:
        cx, cy = center
        if shape == "circle":
            cv2.circle(image, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(mask, (cx, cy), radius, 255, -1, lineType=cv2.LINE_8)
            return
        if shape == "ellipse":
            axes = (max(9, int(radius * 1.15)), max(7, int(radius * 0.63)))
            cv2.ellipse(image, (cx, cy), axes, rotation_deg, 0, 360, color, -1, cv2.LINE_AA)
            cv2.ellipse(mask, (cx, cy), axes, rotation_deg, 0, 360, 255, -1, cv2.LINE_8)
            return

        if shape == "square":
            base = [(-radius, -radius), (radius, -radius), (radius, radius), (-radius, radius)]
        elif shape == "rectangle":
            rx, ry = int(radius * 1.25), int(radius * 0.68)
            base = [(-rx, -ry), (rx, -ry), (rx, ry), (-rx, ry)]
        elif shape == "triangle":
            base = [
                (0, -int(radius * 1.18)),
                (int(radius * 1.05), int(radius * 0.86)),
                (-int(radius * 1.05), int(radius * 0.86)),
            ]
        else:
            raise ValueError(f"unknown shape {shape!r}")

        theta = math.radians(rotation_deg)
        ct, st = math.cos(theta), math.sin(theta)
        pts = []
        for x, y in base:
            rx = x * ct - y * st
            ry = x * st + y * ct
            pts.append((int(round(cx + rx)), int(round(cy + ry))))
        arr = np.asarray(pts, dtype=np.int32)
        cv2.fillPoly(image, [arr], color, lineType=cv2.LINE_AA)
        cv2.fillPoly(mask, [arr], 255, lineType=cv2.LINE_8)

    def generate(
        self,
        color: Optional[str] = None,
        shape: Optional[str] = None,
        difficulty: float = 0.35,
        template: Optional[str] = None,
        add_distractors: bool = True,
    ) -> GroundedEpisode:
        color = color or self.rng.choice(self.color_words)
        shape = shape or self.rng.choice(self.shape_words)
        if color not in self.COLORS:
            raise ValueError(f"unknown teacher color {color}")
        if shape not in self.SHAPES:
            raise ValueError(f"unknown teacher shape {shape}")
        difficulty = float(np.clip(difficulty, 0.0, 1.0))

        image = self._background(difficulty)
        mask = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        radius = self.rng.randint(22, 38)
        margin = radius + 18
        cx = self.rng.randint(margin, self.canvas_size - margin)
        cy = self.rng.randint(margin, self.canvas_size - margin)
        rotation = self.rng.uniform(0, 180) if shape != "circle" else 0.0
        actual_color = self._adjust_color(color, difficulty)

        if add_distractors and difficulty >= 0.25 and self.rng.random() < 0.55:
            distractor_count = 1 if difficulty < 0.7 else 2
            for _ in range(distractor_count):
                dc = self.rng.choice(self.color_words)
                ds = self.rng.choice(self.shape_words)
                dr = self.rng.randint(10, 18)
                for _attempt in range(20):
                    dx = self.rng.randint(dr + 4, self.canvas_size - dr - 4)
                    dy = self.rng.randint(dr + 4, self.canvas_size - dr - 4)
                    if (dx - cx) ** 2 + (dy - cy) ** 2 > (radius + dr + 18) ** 2:
                        break
                temp_mask = np.zeros_like(mask)
                self._draw_shape(
                    image, temp_mask, ds, (dx, dy), dr,
                    self._adjust_color(dc, difficulty), self.rng.uniform(0, 180),
                )

        self._draw_shape(image, mask, shape, (cx, cy), radius, actual_color, rotation)

        if difficulty > 0.4:
            k = 3 if difficulty < 0.82 else 5
            if self.rng.random() < 0.55:
                image = cv2.GaussianBlur(image, (k, k), 0)
            noise = self.nprng.normal(0, 2.0 + 5.0 * difficulty, size=image.shape)
            image = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)

        template = template or self.rng.choice(self.TEMPLATES)
        utterance = template.format(color=color, shape=shape)
        return GroundedEpisode(
            image=image,
            attention_mask=mask,
            utterance=utterance,
            teacher_metadata={
                "color": color,
                "shape": shape,
                "difficulty": difficulty,
                "center": [cx, cy],
                "radius": radius,
                "rotation_deg": rotation,
            },
        )
