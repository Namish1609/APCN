from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple
import json

from apcn_v10.query import KnowledgeQueryEngine
from apcn_v15.conversation import ConversationReply
from apcn_v17.session import CognitiveSessionV17

from .concepts import ConceptBindingMemoryV18, OnlinePrototypeMemoryV18
from .conversation import NaturalConversationEngineV18
from .realizer import AnswerPlanV18, NaturalRealizationMemoryV18, NaturalRealizerV18


class CognitiveSessionV18(CognitiveSessionV17):
    VERSION = "0.18.0"

    def __init__(self, seed: int = 18):
        super().__init__(seed)
        self.seed = seed
        self.v18_binding_repairs = self._repair_knowledge_bindings()
        self.realization_memory_v18 = NaturalRealizationMemoryV18(max_patterns=512)
        self.concept_bindings_v18 = ConceptBindingMemoryV18(max_bindings=4096)
        self.online_prototypes_v18 = OnlinePrototypeMemoryV18(dimensions=256, max_concepts=8192)
        self.v18_concept_sync = self._sync_concept_bindings_from_lexicon()
        self._seed_online_prototypes_from_concepts()
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

    def _sync_concept_bindings_from_lexicon(self) -> Dict[str, object]:
        """Mirror inspectable V0.15 aliases into V0.18 concept identity memory.

        The older lexicon remains the language-facing lexical record. V0.18 adds
        a separate canonical concept-identity graph so semantic clauses can be
        normalized before truth/reasoning operations.
        """
        added: List[Dict[str, object]] = []
        skipped = 0
        for alias, old in sorted(self.lexicon_v15.aliases.items()):
            target = self.lexicon_v15.resolve(old.target)[0]
            if not alias or not target or alias == target:
                skipped += 1
                continue
            current = self.concept_bindings_v18.bindings.get(alias)
            if current is not None and self.concept_bindings_v18.resolve(alias) == self.concept_bindings_v18.resolve(target):
                continue
            try:
                rec = self.concept_bindings_v18.bind(alias, target, source=old.source, weight=max(1, int(old.support)))
            except ValueError:
                skipped += 1
                continue
            self.online_prototypes_v18.bind_alias(alias, target, self.concept_bindings_v18)
            added.append(rec.to_dict())
        return {"mirrored": len(added), "skipped": skipped, "bindings": added[-16:]}

    def _seed_online_prototypes_from_concepts(self) -> None:
        for name, rec in self.concepts.records.items():
            self.online_prototypes_v18.observe_definition(
                name,
                rec.dependencies(),
                kind=rec.kind,
                bindings=self.concept_bindings_v18,
            )

    def _observe_reply_learning(self, reply: ConversationReply, *, legacy_teaching: bool) -> None:
        if not reply.learned:
            return
        if legacy_teaching:
            self.v18_concept_sync = self._sync_concept_bindings_from_lexicon()
            if reply.concept:
                rec = self.concepts.records.get(reply.concept)
                if rec is not None:
                    self.online_prototypes_v18.observe_definition(
                        rec.name,
                        rec.dependencies(),
                        kind=rec.kind,
                        bindings=self.concept_bindings_v18,
                    )
            return
        clause = self.natural_conversation_v18.last_clause
        if clause is not None:
            canonical = self.concept_bindings_v18.canonicalize_clause(clause)
            self.online_prototypes_v18.observe_clause(canonical, self.concept_bindings_v18)

    def _surface_for_v18(self, text: str, *, legacy_teaching: bool) -> Tuple[str, Dict[str, object]]:
        """Apply only explicit concept identity links before semantic parsing.

        Approximate vocabulary repair is intentionally excluded. The existing
        natural conversation layer still asks for clarification on near misses.
        Definition/about queries already resolve via the V0.15 lexical memory, so
        they keep the user's surface term for natural replies.
        """
        if legacy_teaching:
            return text, {"changed": False, "replacements": []}
        q = str(text).strip()
        engine = self.natural_conversation_v18
        if engine.DEFINITION.match(q) or engine.ABOUT.match(q):
            return text, {"changed": False, "replacements": []}
        row = self.concept_bindings_v18.canonicalize_surface(text)
        return str(row["surface"]), row

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
        obj.concept_bindings_v18 = ConceptBindingMemoryV18(max_bindings=4096)
        obj.online_prototypes_v18 = OnlinePrototypeMemoryV18(dimensions=256, max_concepts=8192)
        obj.v18_concept_sync = obj._sync_concept_bindings_from_lexicon()
        obj._seed_online_prototypes_from_concepts()
        obj.v18_language_history = []
        obj.seed = seed
        obj._rebuild_v18_language()
        return obj

    def talk(self, text: str) -> ConversationReply:
        legacy_teaching = self.natural_conversation_v18._is_legacy_explicit_teaching(str(text).strip())
        surface, canonicalization = self._surface_for_v18(text, legacy_teaching=legacy_teaching)
        reply = self.natural_conversation_v18.respond(surface)
        self._observe_reply_learning(reply, legacy_teaching=legacy_teaching)
        trace = list(reply.trace)
        if canonicalization.get("changed"):
            trace.append("v018:concept_identity:canonicalized_surface")
            reply.trace = trace
        self.v18_language_history.append({
            "kind": "conversation",
            "act": reply.act,
            "confidence": reply.confidence,
            "learned": reply.learned,
            "concept": reply.concept,
            "used_v018": any(str(x).startswith("v018:") and "fallback" not in str(x) for x in reply.trace),
            "concept_identity_applied": bool(canonicalization.get("changed")),
        })
        if len(self.v18_language_history) > 4096:
            del self.v18_language_history[: len(self.v18_language_history) - 4096]
        return reply

    def teach_concept_binding(self, alias: str, target: str, *, source: str = "user") -> Dict[str, object]:
        old = self.lexicon_v15.teach_alias(alias, target, source=source)
        rec = self.concept_bindings_v18.bind(alias, old.target, source=source, weight=max(1, int(old.support)))
        self.online_prototypes_v18.bind_alias(alias, old.target, self.concept_bindings_v18)
        return {
            "learned": True,
            "binding": rec.to_dict(),
            "canonical": self.concept_bindings_v18.resolve(alias),
            "prototype": self.online_prototypes_v18.summary(8),
        }

    def concept_similarity(self, a: str, b: str) -> float:
        return self.online_prototypes_v18.similarity(a, b, self.concept_bindings_v18)

    def nearest_concepts(self, concept: str, limit: int = 8) -> List[Dict[str, object]]:
        return self.online_prototypes_v18.nearest(concept, bindings=self.concept_bindings_v18, limit=limit)

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
            "concept_binding_sync": dict(self.v18_concept_sync),
            "concept_bindings": self.concept_bindings_v18.summary(20),
            "online_prototypes": self.online_prototypes_v18.summary(20),
            "realization": self.realization_memory_v18.summary(20),
            "conversation": self.natural_conversation_v18.summary(),
            "architecture_contract": {
                "ordinary_truth_questions_use_structured_semantics": True,
                "structured_semantic_memory_is_truth_authority": True,
                "legacy_property_to_is_a_bridge": False,
                "concept_store_binding_is_canonical": getattr(self.definitions, "store", None) is self.concepts,
                "explicit_aliases_have_concept_identity_layer": True,
                "online_prototypes_are_truth_authority": False,
                "online_prototypes_use_pretrained_embeddings": False,
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
        bindings = out / "concept_bindings_v0_18.json"
        prototypes = out / "online_prototypes_v0_18.json"
        state = out / "session_v0_18.json"
        self.realization_memory_v18.save(realization)
        self.concept_bindings_v18.save(bindings)
        self.online_prototypes_v18.save(prototypes)
        state.write_text(json.dumps({
            "version": self.VERSION,
            "seed": self.seed,
            "v18_language_history": self.v18_language_history,
            "memory_audit": self.v18_memory_audit(),
        }, indent=2), encoding="utf-8")
        return {
            "base_v17": str(base),
            "realization": str(realization),
            "concept_bindings": str(bindings),
            "online_prototypes": str(prototypes),
            "session": str(state),
        }

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
        bindings = out / "concept_bindings_v0_18.json"
        prototypes = out / "online_prototypes_v0_18.json"
        obj.realization_memory_v18 = NaturalRealizationMemoryV18.load(realization) if realization.exists() else NaturalRealizationMemoryV18(max_patterns=512)
        obj.concept_bindings_v18 = ConceptBindingMemoryV18.load(bindings) if bindings.exists() else ConceptBindingMemoryV18(max_bindings=4096)
        obj.online_prototypes_v18 = OnlinePrototypeMemoryV18.load(prototypes) if prototypes.exists() else OnlinePrototypeMemoryV18(dimensions=256, max_concepts=8192)
        obj.v18_concept_sync = obj._sync_concept_bindings_from_lexicon()
        if not prototypes.exists():
            obj._seed_online_prototypes_from_concepts()
        state = out / "session_v0_18.json"
        obj.v18_language_history = []
        if state.exists():
            data = json.loads(state.read_text(encoding="utf-8"))
            obj.v18_language_history = list(data.get("v18_language_history", []))[-4096:]
        obj.seed = seed
        obj.v18_binding_repairs = obj._repair_knowledge_bindings()
        obj._rebuild_v18_language()
        return obj
