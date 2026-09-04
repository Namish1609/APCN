from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import math
import random

import cv2
import numpy as np


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class SyntheticInstanceSpec:
    name: str
    color: str
    shape: str
    texture_seed: int
    category: Tuple[str, ...] = ()


@dataclass
class TemporalFrame:
    image: np.ndarray
    attention_mask: np.ndarray
    bbox: BBox
    teacher_instance: str
    category: Tuple[str, ...]
    visible_fraction: float
    occluder: Optional[BBox] = None


class TemporalSceneTeacher:
    """Controlled temporal visual generator used only for V0.13 evaluation.

    Instance identity is encoded only through ordinary visual micro-texture and
    appearance variation. The learner never receives `teacher_instance` during
    read-only re-identification tests.
    """

    COLORS: Dict[str, Tuple[int, int, int]] = {
        "yellow": (35, 210, 235),
        "red": (50, 60, 215),
        "green": (70, 180, 75),
        "blue": (205, 100, 55),
        "purple": (170, 75, 180),
        "orange": (35, 135, 235),
    }

    def __init__(self, seed: int = 13, width: int = 176, height: int = 128):
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.width = int(width)
        self.height = int(height)

    @staticmethod
    def _draw_shape(mask: np.ndarray, shape: str) -> None:
        h, w = mask.shape
        c = (w//2, h//2)
        shape = shape.lower()
        if shape == "circle":
            cv2.circle(mask, c, int(.34*min(h, w)), 255, -1, lineType=cv2.LINE_AA)
        elif shape == "ellipse":
            cv2.ellipse(mask, c, (int(.36*w), int(.25*h)), 0, 0, 360, 255, -1, lineType=cv2.LINE_AA)
        elif shape == "triangle":
            pts = np.asarray([[w//2, int(.14*h)], [int(.15*w), int(.82*h)], [int(.85*w), int(.82*h)]], np.int32)
            cv2.fillConvexPoly(mask, pts, 255, lineType=cv2.LINE_AA)
        elif shape == "square":
            s = int(.68*min(h, w)); x = (w-s)//2; y=(h-s)//2
            cv2.rectangle(mask, (x,y), (x+s,y+s), 255, -1, lineType=cv2.LINE_AA)
        else:  # rectangle
            cv2.rectangle(mask, (int(.13*w), int(.28*h)), (int(.87*w), int(.72*h)), 255, -1, lineType=cv2.LINE_AA)

    @staticmethod
    def _texture_layer(size: int, seed: int, base: Tuple[int, int, int]) -> np.ndarray:
        rng = np.random.default_rng(seed)
        layer = np.zeros((size, size, 3), dtype=np.uint8)
        layer[:] = np.asarray(base, dtype=np.uint8)
        # Identity-specific but semantically unlabeled microtexture. Several
        # randomized strokes/dots survive viewpoint/brightness changes.
        for _ in range(9):
            x1, y1 = rng.integers(8, size-8, size=2)
            angle = float(rng.uniform(0, math.pi))
            length = int(rng.integers(size//6, size//3))
            x2 = int(np.clip(x1 + math.cos(angle)*length, 2, size-3))
            y2 = int(np.clip(y1 + math.sin(angle)*length, 2, size-3))
            delta = rng.integers(-45, 46, size=3)
            color = tuple(int(x) for x in np.clip(np.asarray(base)+delta, 0, 255))
            cv2.line(layer, (int(x1), int(y1)), (x2,y2), color, int(rng.integers(1,3)), cv2.LINE_AA)
        for _ in range(7):
            x,y = rng.integers(8, size-8, size=2)
            delta = rng.integers(-55, 56, size=3)
            color = tuple(int(x) for x in np.clip(np.asarray(base)+delta, 0, 255))
            cv2.circle(layer, (int(x),int(y)), int(rng.integers(1,4)), color, -1, cv2.LINE_AA)
        return layer

    def render(self, spec: SyntheticInstanceSpec, *, center: Tuple[float,float] = (.5,.5),
               scale: float = .33, angle: float = 0.0, brightness: float = 1.0,
               background_seed: Optional[int] = None, occluder: Optional[BBox] = None) -> TemporalFrame:
        bg_rng = np.random.default_rng(self.seed if background_seed is None else background_seed)
        yy = np.linspace(0, 1, self.height, dtype=np.float64)[:,None]
        xx = np.linspace(0, 1, self.width, dtype=np.float64)[None,:]
        base = 120 + 22*xx + 18*yy
        image = np.dstack([base+8, base, base-7])
        noise = bg_rng.normal(0, 3.2, size=image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)

        psize = 80
        patch_mask = np.zeros((psize,psize), dtype=np.uint8)
        self._draw_shape(patch_mask, spec.shape)
        base_color = np.asarray(self.COLORS[spec.color], dtype=np.float64) * float(brightness)
        base_color = tuple(int(x) for x in np.clip(base_color, 0, 255))
        patch = self._texture_layer(psize, spec.texture_seed, base_color)

        rot = cv2.getRotationMatrix2D((psize/2,psize/2), float(angle), 1.0)
        patch = cv2.warpAffine(patch, rot, (psize,psize), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        patch_mask = cv2.warpAffine(patch_mask, rot, (psize,psize), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        side = max(18, int(scale*min(self.width,self.height)))
        patch = cv2.resize(patch, (side,side), interpolation=cv2.INTER_AREA)
        patch_mask = cv2.resize(patch_mask, (side,side), interpolation=cv2.INTER_AREA)
        cx = int(center[0]*self.width); cy = int(center[1]*self.height)
        x0 = int(np.clip(cx-side//2, 0, self.width-side)); y0 = int(np.clip(cy-side//2, 0, self.height-side))
        x1, y1 = x0+side, y0+side
        roi = image[y0:y1,x0:x1]
        alpha = (patch_mask.astype(np.float64)/255.0)[...,None]
        image[y0:y1,x0:x1] = np.clip(alpha*patch + (1-alpha)*roi, 0, 255).astype(np.uint8)
        full_mask = np.zeros((self.height,self.width), dtype=np.uint8)
        full_mask[y0:y1,x0:x1] = patch_mask
        before = max(1, int(np.count_nonzero(full_mask>32)))

        if occluder is not None:
            ox, oy, ow, oh = occluder
            ax0 = int(np.clip(ox*self.width,0,self.width)); ay0=int(np.clip(oy*self.height,0,self.height))
            ax1 = int(np.clip((ox+ow)*self.width,0,self.width)); ay1=int(np.clip((oy+oh)*self.height,0,self.height))
            image[ay0:ay1,ax0:ax1] = (92,96,100)
            full_mask[ay0:ay1,ax0:ax1] = 0
        after = int(np.count_nonzero(full_mask>32))
        bbox = (x0/self.width, y0/self.height, side/self.width, side/self.height)
        return TemporalFrame(image, full_mask, bbox, spec.name,
                             tuple(spec.category), after/before, occluder)

    def similar_pair(self) -> Tuple[SyntheticInstanceSpec,SyntheticInstanceSpec]:
        # Same category/color/shape; only ordinary appearance details differ.
        a = SyntheticInstanceSpec("milo_bottle", "blue", "rectangle", 13031, ("bottle",))
        b = SyntheticInstanceSpec("twin_bottle", "blue", "rectangle", 13079, ("bottle",))
        return a,b
