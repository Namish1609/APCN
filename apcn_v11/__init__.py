"""APCN V0.11 — unified concept memory, discourse identity and consolidation."""

from .concept_graph import UnifiedConceptGraph, ConceptNode, ConceptEdge
from .error_memory import ErrorMemory, ErrorSignature
from .consolidation import ConsolidationEngine, LearningPrescription
from .discourse import DiscourseEntityRegistry, DiscourseEntity
from .language import SemanticLanguageLearnerV11, AdaptiveLanguageSessionV11
from .testing import run_generated_language_test_v11, semantic_equal_instances
from .session import CognitiveSessionV11

__all__ = [
    "UnifiedConceptGraph", "ConceptNode", "ConceptEdge",
    "ErrorMemory", "ErrorSignature",
    "ConsolidationEngine", "LearningPrescription",
    "DiscourseEntityRegistry", "DiscourseEntity",
    "SemanticLanguageLearnerV11", "AdaptiveLanguageSessionV11",
    "run_generated_language_test_v11", "semantic_equal_instances",
    "CognitiveSessionV11",
]
