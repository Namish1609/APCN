from __future__ import annotations

from pathlib import Path
from typing import Dict
import json

from apcn_v15.conversation import ConversationReply
from apcn_v16.session import CognitiveSessionV16

from .dialogue import StructuredConversationEngineV17
from .language import OperatorConstructionMemoryV17, StructuredLanguageV17, StructuredTeacherV17
from .memory import SemanticDiscourseV17, StructuredSemanticMemoryV17
from .semantics import SemanticClause


class CognitiveSessionV17(CognitiveSessionV16):
    VERSION = "0.17.0"

    def __init__(self, seed: int = 17):
        super().__init__(seed)
        self.seed = seed
        self.operator_memory_v17 = OperatorConstructionMemoryV17(max_patterns=1024)
        self.structured_teacher_v17 = StructuredTeacherV17()
        self.structured_bootstrap_v17 = self.structured_teacher_v17.bootstrap(self.operator_memory_v17, repeats=4)
        self.semantic_memory_v17 = StructuredSemanticMemoryV17(max_records=4096)
        self.discourse_v17 = SemanticDiscourseV17(max_entities=64)
        self.v17_language_history = []
        self._rebuild_v17_language()

    def _rebuild_v17_language(self) -> None:
        self.structured_language_v17 = StructuredLanguageV17(self.operator_memory_v17)
        self.structured_conversation_v17 = StructuredConversationEngineV17(
            self.structured_language_v17,
            self.semantic_memory_v17,
            self.discourse_v17,
            fallback_engine=self.language_v16,
        )

    @staticmethod
    def _adopt_v16_state(obj: "CognitiveSessionV17", old: CognitiveSessionV16) -> None:
        # V0.16 already owns the complete historical cognitive substrate. Copy
        # that state, then rebuild bound language engines so they reference the
        # adopted memories rather than the temporary constructor memories.
        for name, value in old.__dict__.items():
            setattr(obj, name, value)
        obj._rebuild_v16_language()
        obj._rebuild_v17_language()

    @classmethod
    def from_v16_checkpoint(cls, output_dir: str | Path = "outputs/v0_16", *, seed: int = 17) -> "CognitiveSessionV17":
        old = CognitiveSessionV16.load_checkpoint(output_dir, seed=seed)
        obj = cls(seed)
        cls._adopt_v16_state(obj, old)
        # _adopt_v16_state copies only V0.16 fields from old; restore fresh V0.17
        # memories created by the constructor.
        obj.operator_memory_v17 = OperatorConstructionMemoryV17(max_patterns=1024)
        obj.structured_teacher_v17 = StructuredTeacherV17()
        obj.structured_bootstrap_v17 = obj.structured_teacher_v17.bootstrap(obj.operator_memory_v17, repeats=4)
        obj.semantic_memory_v17 = StructuredSemanticMemoryV17(max_records=4096)
        obj.discourse_v17 = SemanticDiscourseV17(max_entities=64)
        obj.v17_language_history = []
        obj.seed = seed
        obj._rebuild_v17_language()
        return obj

    def talk(self, text: str) -> ConversationReply:
        reply = self.structured_conversation_v17.respond(text)
        self.v17_language_history.append({
            "kind": "conversation",
            "act": reply.act,
            "confidence": reply.confidence,
            "learned": reply.learned,
            "concept": reply.concept,
            "used_v017": any(str(x).startswith("v017:") and "fallback" not in str(x) for x in reply.trace),
        })
        if len(self.v17_language_history) > 4096:
            del self.v17_language_history[: len(self.v17_language_history) - 4096]
        return reply

    def parse_structured(self, text: str) -> Dict[str, object]:
        clause, confidence, evidence = self.structured_language_v17.parse(text, discourse=self.discourse_v17)
        if clause is not None:
            self.discourse_v17.ingest(clause)
        return {
            "surface": text,
            "semantic": None if clause is None else clause.to_dict(),
            "confidence": confidence,
            "evidence": evidence,
        }

    def generate_structured(self, clause: SemanticClause, limit: int = 12) -> Dict[str, object]:
        return {
            "semantic": clause.to_dict(),
            "surfaces": self.structured_language_v17.generate(clause, limit),
            "roundtrip": self.structured_language_v17.roundtrip(clause, limit),
        }

    def teach_structured_clause(self, clause: SemanticClause, source: str = "user") -> Dict[str, object]:
        rec = self.semantic_memory_v17.teach(clause, source=source)
        self.discourse_v17.ingest(clause)
        return {"learned": True, "record": rec.to_dict(), "memory": self.semantic_memory_v17.summary(12)}

    def teach_operator_construction(self, op: str, template: str, weight: int = 4) -> Dict[str, object]:
        rec = self.operator_memory_v17.observe(op, template, weight)
        return {"learned": True, "construction": rec.to_dict(), "memory": self.operator_memory_v17.summary(12)}

    def v17_memory_audit(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "operator_constructions": self.operator_memory_v17.summary(20),
            "bootstrap": self.structured_bootstrap_v17,
            "semantic_memory": self.semantic_memory_v17.summary(20),
            "discourse": self.discourse_v17.summary(),
            "language": self.structured_language_v17.summary(),
            "conversation": self.structured_conversation_v17.summary(),
            "architecture_contract": {
                "concept_world_memory_remains_authoritative_for_v016_knowledge": True,
                "v017_structured_facts_are_explicit_semantic_records": True,
                "same_operator_construction_memory_parses_and_generates": True,
                "surface_generator_has_truth_memory_reference": False,
                "raw_chat_transcript_persisted": False,
                "raw_corpus_language_model": False,
                "external_llm": False,
                "neural_language_model": False,
                "visual_training_budget": 0.0,
            },
        }

    def memory_audit(self) -> Dict[str, object]:
        base = super().memory_audit()
        base["v017_structured_semantics"] = self.v17_memory_audit()
        return base

    def save(self, output_dir: str | Path = "outputs/v0_17") -> Dict[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        base = out / "base_v16"
        super().save(base)
        operators = out / "operator_constructions_v0_17.json"
        semantics = out / "semantic_memory_v0_17.json"
        discourse = out / "semantic_discourse_v0_17.json"
        state = out / "session_v0_17.json"
        self.operator_memory_v17.save(operators)
        self.semantic_memory_v17.save(semantics)
        self.discourse_v17.save(discourse)
        state.write_text(json.dumps({
            "version": self.VERSION,
            "seed": self.seed,
            "v17_language_history": self.v17_language_history,
            "memory_audit": self.v17_memory_audit(),
        }, indent=2), encoding="utf-8")
        return {
            "base_v16": str(base),
            "operators": str(operators),
            "semantic_memory": str(semantics),
            "discourse": str(discourse),
            "session": str(state),
        }

    @classmethod
    def load_checkpoint(cls, output_dir: str | Path = "outputs/v0_17", *, seed: int = 17) -> "CognitiveSessionV17":
        out = Path(output_dir)
        base = out / "base_v16"
        if not base.exists():
            raise FileNotFoundError(f"missing V0.17 base checkpoint: {base}")
        old = CognitiveSessionV16.load_checkpoint(base, seed=seed)
        obj = cls(seed)
        cls._adopt_v16_state(obj, old)
        operators = out / "operator_constructions_v0_17.json"
        semantics = out / "semantic_memory_v0_17.json"
        discourse = out / "semantic_discourse_v0_17.json"
        if operators.exists():
            obj.operator_memory_v17 = OperatorConstructionMemoryV17.load(operators)
        else:
            obj.operator_memory_v17 = OperatorConstructionMemoryV17(max_patterns=1024)
            StructuredTeacherV17().bootstrap(obj.operator_memory_v17, repeats=4)
        obj.structured_teacher_v17 = StructuredTeacherV17()
        obj.structured_bootstrap_v17 = {
            "loaded_checkpoint": True,
            "recombination_templates_used": 0,
            "observations": obj.operator_memory_v17.observations,
        }
        obj.semantic_memory_v17 = StructuredSemanticMemoryV17.load(semantics) if semantics.exists() else StructuredSemanticMemoryV17()
        obj.discourse_v17 = SemanticDiscourseV17.load(discourse) if discourse.exists() else SemanticDiscourseV17()
        state = out / "session_v0_17.json"
        obj.v17_language_history = []
        if state.exists():
            data = json.loads(state.read_text(encoding="utf-8"))
            obj.v17_language_history = list(data.get("v17_language_history", []))[-4096:]
        obj.seed = seed
        obj._rebuild_v17_language()
        return obj
