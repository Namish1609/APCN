from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict
import json

from apcn_v10.query import KnowledgeQueryEngine
from apcn_v15.conversation import ConversationReply
from apcn_v17.session import CognitiveSessionV17

from .conversation import NaturalConversationEngineV18
from .realizer import AnswerPlanV18, NaturalRealizationMemoryV18, NaturalRealizerV18


class CognitiveSessionV18(CognitiveSessionV17):
    VERSION = "0.18.0"

    def __init__(self, seed: int = 18):
        super().__init__(seed)
        self.seed = seed
        self.v18_binding_repairs = self._repair_knowledge_bindings()
        self.realization_memory_v18 = NaturalRealizationMemoryV18(max_patterns=512)
        self.v18_language_history = []
        self._rebuild_v18_language()

    def _repair_knowledge_bindings(self) -> Dict[str, object]:
        """Make the inherited concept store canonical before any V0.18 query."""
        merged = 0
        replaced = 0
        detached = False
        definitions = getattr(self, "definitions", None)
        other = getattr(definitions, "store", None)
        if other is not None and other is not self.concepts:
            detached = True
            for name, rec in getattr(other, "records", {}).items():
                current = self.concepts.records.get(name)
                if current is None:
                    self.concepts.records[name] = deepcopy(rec)
                    merged += 1
                elif int(getattr(rec, "support", 0)) > int(getattr(current, "support", 0)):
                    self.concepts.records[name] = deepcopy(rec)
                    replaced += 1
            definitions.store = self.concepts
        self.query = KnowledgeQueryEngine(self.concepts)
        if hasattr(self, "_make_conversation"):
            self.conversation = self._make_conversation()
        if hasattr(self, "_rebuild_v16_language"):
            self._rebuild_v16_language()
        if hasattr(self, "_rebuild_v17_language"):
            self._rebuild_v17_language()
        return {
            "detached_definition_store_found": detached,
            "definition_records_merged": merged,
            "definition_records_replaced": replaced,
            "definitions_share_canonical_concept_store": getattr(self.definitions, "store", None) is self.concepts,
            "query_uses_canonical_concept_store": getattr(self.query, "store", None) is self.concepts,
        }

    def _rebuild_v18_language(self) -> None:
        self.natural_realizer_v18 = NaturalRealizerV18(self.realization_memory_v18)
        self.natural_conversation_v18 = NaturalConversationEngineV18(
            self.structured_language_v17,
            self.semantic_memory_v17,
            self.discourse_v17,
            self.natural_realizer_v18,
            self.concepts,
            self.lexicon_v15,
            self.facts_v15,
            legacy_language=self.language_v16,
            fallback_engine=self.structured_conversation_v17,
        )

    @staticmethod
    def _adopt_v17_state(obj: "CognitiveSessionV18", old: CognitiveSessionV17) -> None:
        for name, value in old.__dict__.items():
            setattr(obj, name, value)
        obj.seed = getattr(old, "seed", obj.seed)
        obj.v18_binding_repairs = obj._repair_knowledge_bindings()

    @classmethod
    def from_v17_checkpoint(cls, output_dir: str | Path = "outputs/v0_17", *, seed: int = 18) -> "CognitiveSessionV18":
        old = CognitiveSessionV17.load_checkpoint(output_dir, seed=seed)
        obj = cls(seed)
        cls._adopt_v17_state(obj, old)
        obj.realization_memory_v18 = NaturalRealizationMemoryV18(max_patterns=512)
        obj.v18_language_history = []
        obj.seed = seed
        obj._rebuild_v18_language()
        return obj

    def talk(self, text: str) -> ConversationReply:
        reply = self.natural_conversation_v18.respond(text)
        self.v18_language_history.append({
            "kind": "conversation",
            "act": reply.act,
            "confidence": reply.confidence,
            "learned": reply.learned,
            "concept": reply.concept,
            "used_v018": any(str(x).startswith("v018:") and "fallback" not in str(x) for x in reply.trace),
        })
        if len(self.v18_language_history) > 4096:
            del self.v18_language_history[: len(self.v18_language_history) - 4096]
        return reply

    def teach_realization(self, act: str, template: str, weight: int = 4) -> Dict[str, object]:
        rec = self.realization_memory_v18.observe(act, template, weight=weight)
        return {"learned": True, "realization": rec.to_dict(), "memory": self.realization_memory_v18.summary(16)}

    def generate_answer_variants(self, plan: AnswerPlanV18, limit: int = 12) -> Dict[str, object]:
        rows = self.natural_realizer_v18.variants(plan, limit)
        return {"plan": plan.to_dict(), "surfaces": rows, "count": len(rows)}

    def v18_memory_audit(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "binding_repairs": dict(self.v18_binding_repairs),
            "realization": self.realization_memory_v18.summary(20),
            "conversation": self.natural_conversation_v18.summary(),
            "architecture_contract": {
                "ordinary_truth_questions_use_structured_semantics": True,
                "structured_semantic_memory_is_truth_authority": True,
                "legacy_property_to_is_a_bridge": False,
                "concept_store_binding_is_canonical": getattr(self.definitions, "store", None) is self.concepts,
                "surface_realizer_has_truth_memory_reference": False,
                "surface_realizer_has_concept_memory_reference": False,
                "surface_realizer_has_world_memory_reference": False,
                "raw_chat_transcript_persisted": False,
                "raw_corpus_language_model": False,
                "external_llm": False,
                "neural_language_model": False,
                "visual_training_budget": 0.0,
            },
        }

    def memory_audit(self) -> Dict[str, object]:
        base = super().memory_audit()
        base["v018_natural_semantic_conversation"] = self.v18_memory_audit()
        return base

    def save(self, output_dir: str | Path = "outputs/v0_18") -> Dict[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        base = out / "base_v17"
        super().save(base)
        realization = out / "natural_realization_v0_18.json"
        state = out / "session_v0_18.json"
        self.realization_memory_v18.save(realization)
        state.write_text(json.dumps({
            "version": self.VERSION,
            "seed": self.seed,
            "v18_language_history": self.v18_language_history,
            "memory_audit": self.v18_memory_audit(),
        }, indent=2), encoding="utf-8")
        return {"base_v17": str(base), "realization": str(realization), "session": str(state)}

    @classmethod
    def load_checkpoint(cls, output_dir: str | Path = "outputs/v0_18", *, seed: int = 18) -> "CognitiveSessionV18":
        out = Path(output_dir)
        base = out / "base_v17"
        if not base.exists():
            raise FileNotFoundError(f"missing V0.18 base checkpoint: {base}")
        old = CognitiveSessionV17.load_checkpoint(base, seed=seed)
        obj = cls(seed)
        cls._adopt_v17_state(obj, old)
        realization = out / "natural_realization_v0_18.json"
        obj.realization_memory_v18 = NaturalRealizationMemoryV18.load(realization) if realization.exists() else NaturalRealizationMemoryV18(max_patterns=512)
        state = out / "session_v0_18.json"
        obj.v18_language_history = []
        if state.exists():
            data = json.loads(state.read_text(encoding="utf-8"))
            obj.v18_language_history = list(data.get("v18_language_history", []))[-4096:]
        obj.seed = seed
        obj.v18_binding_repairs = obj._repair_knowledge_bindings()
        obj._rebuild_v18_language()
        return obj
