from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class FeatureLayout:
    dim: int
    groups: Dict[str, Tuple[int, int]]


class SelfOrganizingPatchSensor:
    """Generic visual front-end based on pixels, normalized occupancy and an
    unlabeled online patch codebook.

    The codebook is learned only from focused pixels. It never receives color or
    shape names. Unlike V0.7-V0.11, this sensor does not compute Hu moments,
    circularity, fill ratio or aspect ratio as semantic-style summary features.

    Inductive biases remain explicit: joint-attention masks, translation/scale
    normalization, a coarse pose normalization, local patches and a spatial
    histogram. V0.12 therefore tests *less handcrafted* perception; it is not a
    claim of learning vision from literally zero prior structure.
    """

    VERSION = "APCN-V0.12-SELF-ORGANIZING-PATCH-SENSOR"

    def __init__(self, normalized_size: int = 32, raster_size: int = 10,
                 histogram_bins: int = 8, max_codewords: int = 24,
                 novelty_threshold: float = 0.24):
        self.normalized_size = int(normalized_size)
        self.raster_size = int(raster_size)
        self.histogram_bins = int(histogram_bins)
        self.max_codewords = int(max_codewords)
        self.novelty_threshold = float(novelty_threshold)
        self.codewords: List[np.ndarray] = []
        self.codeword_counts: List[int] = []
        self.patch_updates = 0

        a = 3 * self.histogram_bins
        b = a + self.raster_size * self.raster_size
        c = b + 5 * self.max_codewords
        self.layout = FeatureLayout(c, {
            "pixel_distribution": (0, a),
            "normalized_raster": (a, b),
            "learned_patch_codebook": (b, c),
        })

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

    @staticmethod
    def _principal_angle(mask: np.ndarray) -> Tuple[float, float]:
        ys, xs = np.where(mask > 0)
        if len(xs) < 8:
            return 0.0, 0.0
        x = xs.astype(np.float64) - float(xs.mean())
        y = ys.astype(np.float64) - float(ys.mean())
        cov = np.asarray([[np.mean(x*x), np.mean(x*y)],
                          [np.mean(x*y), np.mean(y*y)]], dtype=np.float64)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals = vals[order]; vec = vecs[:, order[0]]
        anisotropy = float((vals[0] - vals[1]) / max(vals[0] + vals[1], 1e-8))
        angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
        return angle, anisotropy

    def _prepare(self, image: np.ndarray, attention_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected HxWx3 image")
        mask = self._safe_mask(attention_mask)
        work_img = np.ascontiguousarray(image)
        work_mask = mask

        # Generic pose normalization. For nearly isotropic shapes (circle/square)
        # principal orientation is unstable, so leave orientation untouched.
        angle, anisotropy = self._principal_angle(mask)
        if anisotropy >= 0.14:
            ys, xs = np.where(mask > 0)
            center = (float(xs.mean()), float(ys.mean()))
            rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            bg = tuple(int(x) for x in np.median(image.reshape(-1, 3), axis=0))
            work_img = cv2.warpAffine(image, rot, (image.shape[1], image.shape[0]),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=bg)
            work_mask = cv2.warpAffine(mask, rot, (mask.shape[1], mask.shape[0]),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0)

        ys, xs = np.where(work_mask > 0)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        pad = max(2, int(round(0.08 * max(x1-x0, y1-y0))))
        x0 = max(0, x0-pad); x1 = min(work_img.shape[1], x1+pad)
        y0 = max(0, y0-pad); y1 = min(work_img.shape[0], y1+pad)
        crop_i = work_img[y0:y1, x0:x1]
        crop_m = work_mask[y0:y1, x0:x1]

        h, w = crop_m.shape
        side = max(h, w)
        bg = np.median(image.reshape(-1, 3), axis=0).astype(np.uint8)
        square_i = np.empty((side, side, 3), dtype=np.uint8); square_i[:] = bg
        square_m = np.zeros((side, side), dtype=np.uint8)
        oy, ox = (side-h)//2, (side-w)//2
        square_i[oy:oy+h, ox:ox+w] = crop_i
        square_m[oy:oy+h, ox:ox+w] = crop_m
        n = self.normalized_size
        norm_i = cv2.resize(square_i, (n, n), interpolation=cv2.INTER_AREA)
        norm_m = cv2.resize(square_m, (n, n), interpolation=cv2.INTER_AREA)
        return norm_i, norm_m

    def _patches(self, norm_i: np.ndarray, norm_m: np.ndarray):
        n = self.normalized_size
        img = norm_i.astype(np.float64) / 255.0
        m = norm_m.astype(np.float64) / 255.0
        packed = np.dstack([img, m])
        for cy in range(2, n-2, 4):
            for cx in range(2, n-2, 4):
                patch = packed[cy-1:cy+2, cx-1:cx+2]
                if patch[..., 3].max() < 0.04:
                    continue
                q = (0 if cy < n/2 else 2) + (0 if cx < n/2 else 1)
                yield patch.reshape(-1).astype(np.float64), q

    @staticmethod
    def _rms(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((a-b)**2)))

    def learn(self, image: np.ndarray, attention_mask: np.ndarray) -> int:
        """Unsupervised competitive patch learning; returns patch updates."""
        norm_i, norm_m = self._prepare(image, attention_mask)
        updates = 0
        for patch, _ in self._patches(norm_i, norm_m):
            if not self.codewords:
                self.codewords.append(patch.copy()); self.codeword_counts.append(1)
                updates += 1; continue
            rows = [(self._rms(patch, proto), i) for i, proto in enumerate(self.codewords)]
            dist, idx = min(rows)
            if dist > self.novelty_threshold and len(self.codewords) < self.max_codewords:
                self.codewords.append(patch.copy()); self.codeword_counts.append(1)
            else:
                count = self.codeword_counts[idx] + 1
                # Local bounded adaptation. Slot identity stays stable; updates
                # slow rapidly so old descriptors are not continuously redefined.
                lr = min(0.08, 1.0 / (count ** 0.65))
                self.codewords[idx] = (1.0-lr)*self.codewords[idx] + lr*patch
                self.codeword_counts[idx] = count
            updates += 1
        self.patch_updates += updates
        return updates

    def extract(self, image: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Read-only descriptor extraction. This method never changes codebook."""
        norm_i, norm_m = self._prepare(image, attention_mask)
        focus = norm_m > 32
        pix = norm_i[focus]
        if len(pix) < 4:
            pix = norm_i.reshape(-1, 3)

        hist_parts = []
        for ch in range(3):
            hist, _ = np.histogram(pix[:, ch], bins=self.histogram_bins,
                                   range=(0, 256))
            hist = hist.astype(np.float64)
            hist /= max(float(hist.sum()), 1.0)
            hist_parts.append(hist)
        pixel_hist = np.concatenate(hist_parts)

        raster = cv2.resize(norm_m, (self.raster_size, self.raster_size),
                            interpolation=cv2.INTER_AREA).astype(np.float64).reshape(-1) / 255.0

        code_hist = np.zeros((5, self.max_codewords), dtype=np.float64)
        if self.codewords:
            for patch, q in self._patches(norm_i, norm_m):
                idx = min((self._rms(patch, proto), i)
                          for i, proto in enumerate(self.codewords))[1]
                code_hist[0, idx] += 1.0
                code_hist[1+q, idx] += 1.0
            for r in range(code_hist.shape[0]):
                s = float(code_hist[r].sum())
                if s > 0:
                    code_hist[r] /= s

        vec = np.concatenate([pixel_hist, raster, code_hist.reshape(-1)]).astype(np.float64)
        if vec.shape != (self.dim,):
            raise RuntimeError(f"unexpected V0.12 feature dimension {vec.shape}")
        return vec

    def memory_summary(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "feature_dim": self.dim,
            "codewords": len(self.codewords),
            "max_codewords": self.max_codewords,
            "patch_updates": self.patch_updates,
            "codeword_support": list(self.codeword_counts),
            "raw_patches_retained": 0,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "normalized_size": self.normalized_size,
            "raster_size": self.raster_size,
            "histogram_bins": self.histogram_bins,
            "max_codewords": self.max_codewords,
            "novelty_threshold": self.novelty_threshold,
            "patch_updates": self.patch_updates,
            "codewords": [x.tolist() for x in self.codewords],
            "codeword_counts": list(self.codeword_counts),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "SelfOrganizingPatchSensor":
        obj = cls(
            normalized_size=int(data.get("normalized_size", 32)),
            raster_size=int(data.get("raster_size", 10)),
            histogram_bins=int(data.get("histogram_bins", 8)),
            max_codewords=int(data.get("max_codewords", 24)),
            novelty_threshold=float(data.get("novelty_threshold", .24)),
        )
        obj.patch_updates = int(data.get("patch_updates", 0))
        obj.codewords = [np.asarray(x, dtype=np.float64) for x in data.get("codewords", [])]
        obj.codeword_counts = [int(x) for x in data.get("codeword_counts", [])]
        if len(obj.codewords) != len(obj.codeword_counts):
            raise ValueError("invalid V0.12 sensor codebook")
        if len(obj.codewords) > obj.max_codewords:
            raise ValueError("saved codebook exceeds configured maximum")
        return obj
