from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import json
import random

import cv2
import numpy as np

from .face import BBox, SelfFaceMemory


@dataclass
class FaceBenchmarkResult:
    enroll_views: int
    self_acceptance: float
    unknown_rejection_before_negative: float
    unknown_rejection_after_negative: float
    bounded_memory_ok: bool
    raw_frames_retained: int
    details: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


class SyntheticSelfFaceTeacher:
    """Procedural face-LIKE diagnostic, not a real-face accuracy claim.

    The renderer exists only to regression-test invariance, bounded memory and
    open-set correction in CI. Real webcam performance must be measured by the
    user on their own consenting face.

    The returned box follows the transformed face, analogous to a detector or a
    user updating the focus rectangle. The identity system never receives the
    synthetic identity seed. Small deterministic box jitter remains so the test
    is not pixel-perfect crop matching.
    """

    def __init__(self, seed: int = 14141):
        self.rng = random.Random(seed)

    @staticmethod
    def _identity_params(identity_seed: int):
        r = random.Random(identity_seed)
        return {
            "eye_dx": r.randint(28, 39),
            "eye_y": r.randint(73, 88),
            "eye_r": r.randint(5, 9),
            "nose": r.randint(16, 27),
            "mouth_w": r.randint(29, 48),
            "mouth_y": r.randint(137, 151),
            "brow": r.randint(-8, 8),
            "mark_x": r.randint(72, 148),
            "mark_y": r.randint(92, 132),
            "mark_r": r.randint(2, 5),
        }

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray, rng: random.Random) -> BBox:
        ys, xs = np.where(mask > 20)
        h, w = mask.shape
        if len(xs) < 20:
            return (.15, .055, .70, .89)
        x0, x1 = float(xs.min()), float(xs.max() + 1)
        y0, y1 = float(ys.min()), float(ys.max() + 1)
        bw, bh = x1-x0, y1-y0
        # Detector-like margin plus a small amount of localization jitter.
        pad_x, pad_y = .045*bw, .035*bh
        x0 -= pad_x; x1 += pad_x; y0 -= pad_y; y1 += pad_y
        jx = rng.uniform(-.012, .012) * w
        jy = rng.uniform(-.010, .010) * h
        scale = rng.uniform(.985, 1.018)
        cx, cy = (x0+x1)/2 + jx, (y0+y1)/2 + jy
        bw, bh = (x1-x0)*scale, (y1-y0)*scale
        x0 = max(0.0, cx-bw/2); y0 = max(0.0, cy-bh/2)
        x1 = min(float(w), cx+bw/2); y1 = min(float(h), cy+bh/2)
        return (x0/w, y0/h, max(1.0,x1-x0)/w, max(1.0,y1-y0)/h)

    def render(self, identity_seed: int, view_seed: int) -> Tuple[np.ndarray, BBox]:
        p = self._identity_params(identity_seed)
        r = random.Random(view_seed)
        h = w = 220
        bg = r.randint(24, 62)
        img = np.full((h, w, 3), bg, dtype=np.uint8)
        face_mask = np.zeros((h, w), dtype=np.uint8)
        # Neutral synthetic palette; identity signal comes mainly from geometry
        # and local structure, not skin-tone classes.
        base = r.randint(150, 205)
        face_color = (base, min(240, base + 8), min(245, base + 15))
        cv2.ellipse(img, (110, 111), (70, 88), 0, 0, 360, face_color, -1, cv2.LINE_AA)
        cv2.ellipse(img, (42, 112), (10, 24), 0, 0, 360, face_color, -1, cv2.LINE_AA)
        cv2.ellipse(img, (178, 112), (10, 24), 0, 0, 360, face_color, -1, cv2.LINE_AA)
        cv2.ellipse(face_mask, (110, 111), (70, 88), 0, 0, 360, 255, -1, cv2.LINE_AA)
        cv2.ellipse(face_mask, (42, 112), (10, 24), 0, 0, 360, 255, -1, cv2.LINE_AA)
        cv2.ellipse(face_mask, (178, 112), (10, 24), 0, 0, 360, 255, -1, cv2.LINE_AA)
        for sign in (-1, 1):
            ex = 110 + sign*p["eye_dx"]
            cv2.circle(img, (ex, p["eye_y"]), p["eye_r"]+3, (235,235,235), -1, cv2.LINE_AA)
            cv2.circle(img, (ex, p["eye_y"]), p["eye_r"], (35,35,35), -1, cv2.LINE_AA)
            y0 = p["eye_y"] - 17
            cv2.line(img, (ex-13, y0-sign*p["brow"]//3), (ex+13, y0+sign*p["brow"]//3), (55,55,55), 3, cv2.LINE_AA)
        cv2.line(img, (110, 94), (108 + p["brow"]//3, 94+p["nose"]), (90,90,90), 3, cv2.LINE_AA)
        cv2.line(img, (108+p["brow"]//3, 94+p["nose"]), (117, 120), (90,90,90), 2, cv2.LINE_AA)
        cv2.ellipse(img, (110, p["mouth_y"]), (p["mouth_w"], 10+r.randint(-2,3)), 0, 8, 172, (65,65,65), 3, cv2.LINE_AA)
        cv2.circle(img, (p["mark_x"], p["mark_y"]), p["mark_r"], (70,70,70), -1, cv2.LINE_AA)
        rr = random.Random(identity_seed + 1000)
        for _ in range(12):
            x = rr.randint(65, 155); y = rr.randint(62, 154)
            cv2.circle(img, (x, y), 1, (110,110,110), -1)

        # View nuisance: expression, brightness, blur, small pose and shift.
        mouth_shift = r.randint(-4, 4)
        if mouth_shift:
            cv2.line(img, (82, 154+mouth_shift), (138, 154-mouth_shift), (90,90,90), 1, cv2.LINE_AA)
        gain = r.uniform(.72, 1.22); bias = r.uniform(-8, 10)
        img = np.clip(img.astype(np.float32)*gain + bias, 0, 255).astype(np.uint8)
        if r.random() < .35:
            img = cv2.GaussianBlur(img, (3,3), r.uniform(.2, .8))
        angle = r.uniform(-9.0, 9.0); scale = r.uniform(.94, 1.05)
        dx, dy = r.randint(-7,7), r.randint(-5,5)
        M = cv2.getRotationMatrix2D((110,110), angle, scale); M[:,2] += (dx,dy)
        img = cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        face_mask = cv2.warpAffine(face_mask, M, (w,h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img, self._bbox_from_mask(face_mask, r)


def _distribution(rows):
    scores = np.asarray([float(x.get("score", 0.0)) for x in rows], dtype=np.float64)
    appearances = np.asarray([float(x.get("appearance_score", 0.0)) for x in rows], dtype=np.float64)
    states: Dict[str, int] = {}
    for x in rows:
        state = str(x.get("state", "?")); states[state] = states.get(state, 0) + 1
    if len(scores) == 0:
        return {"states": states}
    return {
        "states": states,
        "score_mean": float(scores.mean()),
        "score_min": float(scores.min()),
        "score_p10": float(np.quantile(scores, .10)),
        "score_median": float(np.median(scores)),
        "score_p90": float(np.quantile(scores, .90)),
        "score_max": float(scores.max()),
        "appearance_mean": float(appearances.mean()),
    }


def run_face_benchmark(seed: int = 14141, *, enroll_views: int = 10,
                       test_views: int = 50, negative_examples: int = 4) -> FaceBenchmarkResult:
    teacher = SyntheticSelfFaceTeacher(seed)
    memory = SelfFaceMemory(max_views=12)
    self_seed = seed + 111
    unknown_seed = seed + 9999

    for i in range(enroll_views):
        frame, bbox = teacher.render(self_seed, seed + 100 + i)
        memory.enroll("me", frame, bbox)

    self_rows = []
    for i in range(test_views):
        frame, bbox = teacher.render(self_seed, seed + 10000 + i)
        self_rows.append(memory.recognize(frame, bbox))
    self_ok = sum(int(x.get("match", False)) for x in self_rows)

    unknown_before = []
    for i in range(test_views):
        frame, bbox = teacher.render(unknown_seed, seed + 20000 + i)
        unknown_before.append(memory.recognize(frame, bbox))
    unknown_ok_before = sum(int(not x.get("match", False)) for x in unknown_before)

    negative_rows = []
    for i in range(negative_examples):
        frame, bbox = teacher.render(unknown_seed, seed + 30000 + i)
        negative_rows.append(memory.mark_not_me(frame, bbox))

    unknown_after = []
    for i in range(test_views):
        frame, bbox = teacher.render(unknown_seed, seed + 40000 + i)
        unknown_after.append(memory.recognize(frame, bbox))
    unknown_ok_after = sum(int(not x.get("match", False)) for x in unknown_after)

    summary = memory.summary(); inst = summary["instance_memory"]
    bounded = (
        inst["positive_prototypes"] <= inst["max_views_per_instance"] and
        inst["negative_prototypes"] <= 4 and
        inst["raw_frames_retained"] == 0 and
        inst["raw_descriptors_retained"] == 0 and
        summary["raw_face_images_retained"] == 0 and
        summary["raw_camera_frames_retained"] == 0
    )
    false_negative_states: Dict[str, int] = {}
    for x in self_rows:
        if not x.get("match", False):
            st = str(x.get("state", "?")); false_negative_states[st] = false_negative_states.get(st, 0) + 1
    false_positive_states: Dict[str, int] = {}
    for x in unknown_before:
        if x.get("match", False):
            st = str(x.get("state", "?")); false_positive_states[st] = false_positive_states.get(st, 0) + 1

    return FaceBenchmarkResult(
        enroll_views=enroll_views,
        self_acceptance=self_ok/max(1,test_views),
        unknown_rejection_before_negative=unknown_ok_before/max(1,test_views),
        unknown_rejection_after_negative=unknown_ok_after/max(1,test_views),
        bounded_memory_ok=bool(bounded),
        raw_frames_retained=0,
        details={
            "self_distribution": _distribution(self_rows),
            "unknown_before_distribution": _distribution(unknown_before),
            "unknown_after_distribution": _distribution(unknown_after),
            "false_negative_states": false_negative_states,
            "false_positive_states_before_negative": false_positive_states,
            "negative_corrections": negative_rows,
            "memory": summary,
            "locator_protocol": "face bbox tracks transformed face with small jitter; identity seed is never supplied to recognizer",
            "scientific_boundary": "procedural face-like CI diagnostic only; not real-face or security accuracy",
        },
    )


def main() -> int:
    report = run_face_benchmark()
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
