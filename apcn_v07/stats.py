from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable
import numpy as np


@dataclass
class RunningStats:
    """Compact sufficient statistics. No individual training examples are retained."""

    count: int
    sum: np.ndarray
    sumsq: np.ndarray

    @classmethod
    def empty(cls, dim: int) -> "RunningStats":
        return cls(0, np.zeros(dim, dtype=np.float64), np.zeros(dim, dtype=np.float64))

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != self.sum.shape:
            raise ValueError(f"feature shape mismatch: expected {self.sum.shape}, got {x.shape}")
        self.count += 1
        self.sum += x
        self.sumsq += x * x

    @property
    def mean(self) -> np.ndarray:
        if self.count == 0:
            return np.zeros_like(self.sum)
        return self.sum / float(self.count)

    @property
    def var(self) -> np.ndarray:
        if self.count <= 1:
            return np.ones_like(self.sum) * 1e-6
        m = self.mean
        return np.maximum(self.sumsq / float(self.count) - m * m, 1e-9)

    def complement(self, total: "RunningStats") -> "RunningStats":
        n = total.count - self.count
        if n <= 0:
            return RunningStats.empty(len(self.sum))
        return RunningStats(
            n,
            total.sum - self.sum,
            np.maximum(total.sumsq - self.sumsq, 0.0),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "count": self.count,
            "sum": self.sum.tolist(),
            "sumsq": self.sumsq.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "RunningStats":
        return cls(
            int(data["count"]),
            np.asarray(data["sum"], dtype=np.float64),
            np.asarray(data["sumsq"], dtype=np.float64),
        )
