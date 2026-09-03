"""APCN V0.12 — self-organizing perception research branch."""

from .sensor import SelfOrganizingPatchSensor
from .visual import SelfOrganizingVisualLearner
from .session import TrainingSessionV12, CognitiveSessionV12
from .benchmark import PairedBenchmarkReport, run_paired_benchmark

__all__ = [
    "SelfOrganizingPatchSensor", "SelfOrganizingVisualLearner",
    "TrainingSessionV12", "CognitiveSessionV12",
    "PairedBenchmarkReport", "run_paired_benchmark",
]
