"""APCN V0.12 — self-organizing perception + adaptive correction."""

from .sensor import SelfOrganizingPatchSensor
from .visual import SelfOrganizingVisualLearner
from .language import (
    AdaptiveConstructionCalibrator, SemanticLanguageLearnerV12,
    AdaptiveLanguageSessionV12,
)
from .session import TrainingSessionV12, CognitiveSessionV12
from .benchmark import PairedBenchmarkReport, run_paired_benchmark

__all__ = [
    "SelfOrganizingPatchSensor", "SelfOrganizingVisualLearner",
    "AdaptiveConstructionCalibrator", "SemanticLanguageLearnerV12",
    "AdaptiveLanguageSessionV12", "TrainingSessionV12", "CognitiveSessionV12",
    "PairedBenchmarkReport", "run_paired_benchmark",
]
