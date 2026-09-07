from __future__ import annotations

from pathlib import Path
from typing import Dict
import json

from apcn_v15.session import CognitiveSessionV15
from apcn_v15.conversation import ConversationReply
from .bidirectional import (
    SemanticFrame,
    BidirectionalConstructionMemory,
    BidirectionalTeacherV16,
    SemanticResponderV16,
    BidirectionalLanguageEngineV16,
)


class CognitiveSessionV16(CognitiveSessionV15):
    VERSION = "0.16.0-dev"

    def __init__(self, seed: int = 16):
        super().__init__(seed)
        self.seed = seed
        self.bidirectional_v16 = BidirectionalConstructionMemory(max_patterns=2048)
        self.bidirectional_teacher_v16 = BidirectionalTeacherV16()
        self.bidirectional_bootstrap_v16 = self.bidirectional_teacher_v16.bootstrap(self.bidirectional_v16, repeats=4)
        self.v16_language_history = []
        self._rebuild_v16_language()

    def _rebuild_v16_language(self) -> None:
        self.semantic_responder_v16 = SemanticResponderV16(self.concepts, self.lexicon_v15, self.facts_v15)
        self.language_v16 = BidirectionalLanguageEngineV16(
            self.bidirectional_v16,
            self.semantic_responder_v16,
            fallback_engine=self.conversation,
        )

    @staticmethod
    def _adopt_v15_state(obj: "CognitiveSessionV16", old: CognitiveSessionV15) -> None:
        # Preserve the entire V0.15 cognitive substrate while keeping the V0.16
        # construction memory separate. The language generator does not receive a
        # reference to the concept/world stores; only SemanticResponderV16 does.
        for name in (
            "visual", "concepts", "definitions", "query", "graph", "errors",
            "consolidation", "world", "visual_test_history", "language_test_history",
            "test_history", "consolidation_history", "world_test_history",
            "v012_bootstrap_experiences", "language", "self_face",
            "v14_language_history", "v14_face_history", "lexicon_v15", "facts_v15",
            "dialogue_learner_v15", "dialogue_teacher_v15", "dialogue_bootstrap_v15",
            "v15_language_history", "language_budget_ratio",
        ):
            if hasattr(old, name):
                setattr(obj, name, getattr(old, name))
        obj.conversation = obj._make_conversation()
        obj._rebuild_v16_language()

    @classmethod
    def from_v15_checkpoint(cls, output_dir: str | Path = "outputs/v0_15", *, seed: int = 16) -> "CognitiveSessionV16":
        old = CognitiveSessionV15.load_checkpoint(output_dir, seed=seed)
        obj = cls(seed)
        cls._adopt_v15_state(obj, old)
        return obj

    def talk(self, text: str) -> ConversationReply:
        reply = self.language_v16.respond(text)
        self.v16_language_history.append({
            "kind": "conversation",
            "act": reply.act,
            "confidence": reply.confidence,
            "concept": reply.concept,
            "used_v016": any(str(x).startswith("v016:") and "fallback" not in str(x) for x in reply.trace),
        })
        if len(self.v16_language_history) > 4096:
            del self.v16_language_history[: len(self.v16_language_history) - 4096]
        return reply

    def teach_bidirectional_construction(self, surface: str, frame: SemanticFrame, weight: int = 4) -> Dict[str, object]:
        rec = self.bidirectional_v16.observe(surface, frame, weight=weight)
        return {"learned": True, "construction": rec.to_dict(), "memory": self.bidirectional_v16.summary(12)}

    def paraphrase(self, text: str, limit: int = 10) -> Dict[str, object]:
        return self.language_v16.paraphrases(text, limit)

    def generate_from_semantics(self, frame: SemanticFrame, limit: int = 12) -> Dict[str, object]:
        rows = self.bidirectional_v16.generate(frame, limit)
        return {
            "semantic": frame.to_dict(),
            "surfaces": rows,
            "count": len(rows),
            "roundtrip": self.bidirectional_v16.roundtrip(frame, limit),
        }

    def v16_memory_audit(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "bidirectional_constructions": self.bidirectional_v16.summary(20),
            "bootstrap": self.bidirectional_bootstrap_v16,
            "engine": self.language_v16.summary(),
            "architecture_contract": {
                "same_construction_memory_used_for_parse_and_generate": True,
                "concept_world_memory_is_authoritative": True,
                "semantic_responder_is_truth_firewall": True,
                "surface_generator_has_world_memory_reference": False,
                "raw_corpus_language_model": False,
                "external_llm": False,
                "neural_language_model": False,
            },
        }

    def memory_audit(self) -> Dict[str, object]:
        base = super().memory_audit()
        base["v016_bidirectional_language"] = self.v16_memory_audit()
        return base

    def save(self, output_dir: str | Path = "outputs/v0_16") -> Dict[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        base = out / "base_v15"
        super().save(base)
        constructions = out / "bidirectional_language_v0_16.json"
        state = out / "session_v0_16.json"
        self.bidirectional_v16.save(constructions)
        state.write_text(json.dumps({
            "version": self.VERSION,
            "seed": self.seed,
            "v16_language_history": self.v16_language_history,
            "memory_audit": self.v16_memory_audit(),
        }, indent=2), encoding="utf-8")
        return {"base_v15": str(base), "constructions": str(constructions), "session": str(state)}

    @classmethod
    def load_checkpoint(cls, output_dir: str | Path = "outputs/v0_16", *, seed: int = 16) -> "CognitiveSessionV16":
        out = Path(output_dir)
        base = out / "base_v15"
        if not base.exists():
            raise FileNotFoundError(f"missing V0.16 base checkpoint: {base}")
        old = CognitiveSessionV15.load_checkpoint(base, seed=seed)
        obj = cls(seed)
        cls._adopt_v15_state(obj, old)
        constructions = out / "bidirectional_language_v0_16.json"
        if constructions.exists():
            obj.bidirectional_v16 = BidirectionalConstructionMemory.load(constructions)
        state = out / "session_v0_16.json"
        if state.exists():
            data = json.loads(state.read_text(encoding="utf-8"))
            obj.v16_language_history = list(data.get("v16_language_history", []))[-4096:]
        obj.bidirectional_teacher_v16 = BidirectionalTeacherV16()
        obj.bidirectional_bootstrap_v16 = {
            "loaded_checkpoint": True,
            "dev_templates_used": 0,
            "observations": obj.bidirectional_v16.observations,
        }
        obj._rebuild_v16_language()
        return obj
