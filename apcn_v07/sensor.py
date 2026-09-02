from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class FeatureLayout:
    dim: int
    groups: Dict[str, Tuple[int, int]]


class AnonymousVisualSensor:
    """
    Converts focused pixels into an anonymous numeric vector.

    Important boundary:
      * The learner sees only f000, f001, ... dimensions.
      * It is NOT told which dimensions are chromatic, geometric, positional, etc.
      * This V0.7 front-end is still engineered signal processing, not raw-pixel
        feature discovery. The purpose is to test whether language can select the
        relevant subspace through controlled variation.
    """

    def __init__(self):
        self.layout = FeatureLayout(
            dim=23,
            groups={
                "channel_signal": (0, 9),
                "scale_position": (9, 14),
                "geometry_signal": (14, 23),
            },
        )

    @property
    def dim(self) -> int:
        return self.layout.dim

    def feature_ids(self) -> List[str]:
        return [f"f{i:03d}" for i in range(self.dim)]

    @staticmethod
    def _safe_mask(mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3:
            mask = mask[..., 0]
        out = (mask > 0).astype(np.uint8) * 255
        if cv2.countNonZero(out) < 8:
            raise ValueError("attention mask contains too few focused pixels")
        return out

    def extract(self, image: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected HxWx3 uint8 image")
        mask = self._safe_mask(attention_mask)
        h, w = mask.shape

        pix = image[mask > 0].astype(np.float64) / 255.0
        mean_c = pix.mean(axis=0)
        std_c = pix.std(axis=0)
        channel_sum = np.maximum(mean_c.sum(), 1e-6)
        ratios = mean_c / channel_sum

        ys, xs = np.where(mask > 0)
        area = len(xs) / float(h * w)
        cx = float(xs.mean()) / max(w - 1, 1)
        cy = float(ys.mean()) / max(h - 1, 1)
        bw = float(xs.max() - xs.min() + 1) / float(w)
        bh = float(ys.max() - ys.min() + 1) / float(h)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        c_area = max(float(cv2.contourArea(contour)), 1.0)
        _, _, cw, ch = cv2.boundingRect(contour)
        bbox_area = max(float(cw * ch), 1.0)
        fill = np.clip(c_area / bbox_area, 0.0, 1.0)
        aspect = np.tanh(np.log(max(cw, 1) / max(ch, 1)))

        moments = cv2.moments(contour)
        hu = cv2.HuMoments(moments).reshape(-1)
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
        hu_scaled = np.tanh(hu_log / 8.0)

        vec = np.concatenate([
            mean_c,
            std_c,
            ratios,
            np.asarray([area, cx, cy, bw, bh]),
            np.asarray([fill, aspect]),
            hu_scaled,
        ]).astype(np.float64)
        if vec.shape != (self.dim,):
            raise RuntimeError(f"unexpected feature dimension {vec.shape}")
        return vec
