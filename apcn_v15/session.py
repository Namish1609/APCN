from __future__ import annotations

from pathlib import Path
from typing import Dict
import json

from apcn_v14.session import CognitiveSessionV14
from .conversation import ConversationReply
from .corpus import EnglishExposureMemory
from .dialogue_learning import test_dialogue_learner, train_dialogue_learner
from .dialogue_learning_v151 import ConversationTeacherV151, DialogueActLearnerV151
from .learned_conversation import LearnedConversationEngine
from .lexicon import FactMemory, LexicalSemanticMemory


class CognitiveSessionV15(CognitiveSessionV14):
    VERSION = "0.15.0"

    def __init__(self, seed: int = 15):
        super().__init__(seed)
        self.seed = seed
        # V0.15 is intentionally a language-only research release. Existing
        # perception/world memory is preserved as grounding, but automatic
        # development budget no longer adds visual training experiences.
        self.language_budget_ratio = 1.0
        self.lexicon_v15 = LexicalSemanticMemory()
        self.facts_v15 = FactMemory()
        self.english_exposure_v15 = EnglishExposureMemory()
        self.dialogue_learner_v15 = DialogueActLearnerV151()
        self.dialogue_teacher_v15 = ConversationTeacherV151(seed + 15000)
        # Small deterministic bootstrap makes the Conversation tab usable on a
        # fresh install. Subsequent Language Only training continues improving
        # this learned dialogue memory rather than expanding parser regexes.
        train_dialogue_learner(self.dialogue_learner_v15, self.dialogue_teacher_v15, 480)
        self.v15_language_history = []
        self.conversation = self._make_conversation()

    def _make_conversation(self) -> LearnedConversationEngine:
        return LearnedConversationEngine(
            self.concepts,
            self.lexicon_v15,
            self.facts_v15,
            dialogue_learner=self.dialogue_learner_v15,
            semantic_parser=self.language.learner.parse,
            discourse_registry=self.language.discourse,
            world_query=self.where,
        )

    @staticmethod
    def _adopt_v14_state(obj: "CognitiveSessionV15", old: CognitiveSessionV14) -> None:
        # Preserve all previously learned cognition. V0.15 changes only the active
        # research priority and adds conversational memories.
        for name in (
            "visual", "concepts", "definitions", "query", "graph", "errors",
            "consolidation", "world", "visual_test_history", "language_test_history",
            "test_history", "consolidation_history", "world_test_history",
            "v012_bootstrap_experiences", "language", "self_face",
            "v14_language_history", "v14_face_history",
        ):
            if hasattr(old, name):
                setattr(obj, name, getattr(old, name))
        obj.language_budget_ratio = 1.0
        obj.conversation = obj._make_conversation()

    @classmethod
    def from_v14_checkpoint(cls, output_dir: str | Path = "outputs/v0_14", *, seed: int = 15) -> "CognitiveSessionV15":
        old = CognitiveSessionV14.load_checkpoint(output_dir, seed=seed)
        obj = cls(seed)
        cls._adopt_v14_state(obj, old)
        return obj

    def talk(self, text: str) -> ConversationReply:
        reply = self.conversation.respond(text)
        # Persist only a semantic training/audit history, never a raw chat log.
        self.v15_language_history.append({
            "kind": "conversation",
            "act": reply.act,
            "confidence": reply.confidence,
            "learned": reply.learned,
            "concept": reply.concept,
        })
        if len(self.v15_language_history) > 4096:
            del self.v15_language_history[: len(self.v15_language_history) - 4096]
        return reply

    def language_only_train(self, steps: int = 1000) -> Dict[str, object]:
        """Spend the entire V0.15 learning budget on language.

        Roughly 55% of experiences teach general conversational dialogue acts;
        the rest continue V0.14 grounded semantic construction learning. Neither
        path changes visual training state.
        """
        steps = max(2, int(steps))
        dialogue_steps = max(1, int(round(steps * .55)))
        semantic_steps = max(1, steps - dialogue_steps)
        visual_before = self.visual.learner.episode_count
        drow = train_dialogue_learner(self.dialogue_learner_v15, self.dialogue_teacher_v15, dialogue_steps)
        srow = self.language_first_train(semantic_steps)
        visual_after = self.visual.learner.episode_count
        result = {
            "language_only": True,
            "requested_steps": steps,
            "dialogue_steps": dialogue_steps,
            "grounded_semantic_steps": semantic_steps,
            "dialogue_correct_before_learning": drow["correct_before_learning"],
            "grounded_correct_before_learning": srow["correct_before_learning_rate"],
            "experiences_added": drow["steps"] + srow["experiences_added"],
            "visual_experiences_added": visual_after - visual_before,
            "dialogue_memory": self.dialogue_learner_v15.summary(12),
            "grounded_program_constructions": srow["program_constructions"],
        }
        self.v15_language_history.append({
            "kind": "language_train",
            "dialogue_steps": dialogue_steps,
            "grounded_semantic_steps": semantic_steps,
            "dialogue_correct_before_learning": drow["correct_before_learning"],
            "grounded_correct_before_learning": srow["correct_before_learning_rate"],
        })
        self.conversation = self._make_conversation()
        return result

    def test_dialogue_generalization(self, samples: int = 240) -> Dict[str, object]:
        teacher = ConversationTeacherV151(self.seed + 15111)
        return test_dialogue_learner(self.dialogue_learner_v15, teacher, samples)

    def ingest_english_text(self, text: str) -> Dict[str, object]:
        """Expose APCN to English surface statistics without asserting semantics."""
        row = self.english_exposure_v15.ingest(text)
        self.v15_language_history.append({"kind": "english_exposure", **row})
        if len(self.v15_language_history) > 4096:
            del self.v15_language_history[: len(self.v15_language_history) - 4096]
        return {
            **row,
            "semantic_learning": False,
            "raw_text_retained": False,
            "note": "surface familiarity only; semantic meaning still requires grounding/definition/demonstration",
        }

    def english_coverage(self, text: str) -> Dict[str, object]:
        semantic_terms = set(self.concepts.records)
        semantic_terms.update(self.lexicon_v15.aliases)
        semantic_terms.update(self.lexicon_v15.aliases[k].target for k in self.lexicon_v15.aliases)
        semantic_terms.update(r.subject for r in self.facts_v15.facts.values())
        semantic_terms.update(r.object for r in self.facts_v15.facts.values())
        return self.english_exposure_v15.coverage(text, semantic_terms)

    def conversation_memory_audit(self) -> Dict[str, object]:
        return {
            "lexicon": self.lexicon_v15.summary(16),
            "facts": self.facts_v15.summary(16),
            "english_exposure": self.english_exposure_v15.summary(16),
            "dialogue_constructions": self.dialogue_learner_v15.summary(16),
            "conversation": self.conversation.summary(),
            "raw_chat_transcript_persisted": False,
            "language_budget_ratio": 1.0,
        }

    def memory_audit(self) -> Dict[str, object]:
        base = super().memory_audit()
        base["v015_conversation"] = self.conversation_memory_audit()
        return base

    def save(self, output_dir: str | Path = "outputs/v0_15") -> Dict[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        base = out / "base_v14"
        super().save(base)
        lex = out / "lexicon_v0_15.json"
        facts = out / "facts_v0_15.json"
        exposure = out / "english_exposure_v0_15.json"
        dialogue = out / "dialogue_memory_v0_15.json"
        state = out / "session_v0_15.json"
        self.lexicon_v15.save(lex)
        self.facts_v15.save(facts)
        self.english_exposure_v15.save(exposure)
        self.dialogue_learner_v15.save(dialogue)
        state.write_text(json.dumps({
            "version": self.VERSION,
            "seed": self.seed,
            "language_budget_ratio": 1.0,
            "v15_language_history": self.v15_language_history,
            "memory_audit": self.conversation_memory_audit(),
        }, indent=2), encoding="utf-8")
        return {
            "base_v14": str(base),
            "lexicon": str(lex),
            "facts": str(facts),
            "english_exposure": str(exposure),
            "dialogue": str(dialogue),
            "session": str(state),
        }

    @classmethod
    def load_checkpoint(cls, output_dir: str | Path = "outputs/v0_15", *, seed: int = 15) -> "CognitiveSessionV15":
        out = Path(output_dir)
        base = out / "base_v14"
        if not base.exists():
            raise FileNotFoundError(f"missing V0.15 base checkpoint: {base}")
        old = CognitiveSessionV14.load_checkpoint(base, seed=seed)
        obj = cls(seed)
        cls._adopt_v14_state(obj, old)
        lex = out / "lexicon_v0_15.json"
        facts = out / "facts_v0_15.json"
        exposure = out / "english_exposure_v0_15.json"
        dialogue = out / "dialogue_memory_v0_15.json"
        if lex.exists():
            obj.lexicon_v15 = LexicalSemanticMemory.load(lex)
        if facts.exists():
            obj.facts_v15 = FactMemory.load(facts)
        if exposure.exists():
            obj.english_exposure_v15 = EnglishExposureMemory.load(exposure)
        if dialogue.exists():
            obj.dialogue_learner_v15 = DialogueActLearnerV151.load(dialogue)
        state = out / "session_v0_15.json"
        if state.exists():
            data = json.loads(state.read_text(encoding="utf-8"))
            obj.v15_language_history = list(data.get("v15_language_history", []))[-4096:]
        obj.dialogue_teacher_v15 = ConversationTeacherV151(seed + 15000)
        obj.language_budget_ratio = 1.0
        obj.conversation = obj._make_conversation()
        return obj
