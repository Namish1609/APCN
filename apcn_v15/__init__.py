"""APCN V0.15 — Conversational Language Core.

Language-only research milestone: explicit lexical learning, dialogue state,
knowledge teaching, clarification and bounded conversational response planning.
No external LLM is used by this package.
"""

from .conversation import ConversationEngine, ConversationReply, DialogueState
from .lexicon import LexicalSemanticMemory, FactMemory
from .session import CognitiveSessionV15

__all__ = [
    "ConversationEngine",
    "ConversationReply",
    "DialogueState",
    "LexicalSemanticMemory",
    "FactMemory",
    "CognitiveSessionV15",
]
