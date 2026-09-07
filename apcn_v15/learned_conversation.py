from __future__ import annotations

from typing import List, Optional, Tuple
import re

from apcn_v10.definitions import normalize_name
from .conversation import ConversationEngine, ConversationReply
from .dialogue_learning import DialogueActLearner


class LearnedConversationEngine(ConversationEngine):
    """ConversationEngine augmented by learned dialogue-act construction cues.

    V0.15 uses a two-path interpreter:
      1. direct/high-confidence constructions already supported by the explicit
         conversational shell retain priority;
      2. the learned sparse dialogue memory is a FALLBACK for new paraphrases.

    This prevents a weaker learned hypothesis from overriding a construction the
    system already understands reliably, while still allowing unseen surface
    forms to generalize without adding their wording as parser regexes.
    """

    VERSION = "APCN-V0.15-LEARNED-CONVERSATION-ENGINE"

    def __init__(self, *args, dialogue_learner: DialogueActLearner, **kwargs):
        super().__init__(*args, **kwargs)
        self.dialogue_learner = dialogue_learner
        self.last_learned_dialogue_evidence = []

    def _known_terms(self) -> List[str]:
        terms = set(self.concepts.records)
        terms.update(self.lexicon.aliases)
        terms.update(rec.target for rec in self.lexicon.aliases.values())
        for rec in self.facts.facts.values():
            terms.add(rec.subject); terms.add(rec.object)
        return sorted((normalize_name(x) for x in terms if normalize_name(x)), key=len, reverse=True)

    def _mentioned_concepts(self, text: str) -> List[str]:
        lower = " " + normalize_name(text) + " "
        found: List[str] = []
        occupied: List[Tuple[int,int]] = []
        for term in self._known_terms():
            pattern = re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", re.I)
            m = pattern.search(lower)
            if not m:
                continue
            span = (m.start(), m.end())
            if any(not (span[1] <= a or span[0] >= b) for a,b in occupied):
                continue
            occupied.append(span); found.append(term)
        return found

    def _learned_reply(self, text: str) -> Optional[ConversationReply]:
        concepts = self._mentioned_concepts(text)
        act, conf, evidence = self.dialogue_learner.predict(text, concepts)
        self.last_learned_dialogue_evidence = evidence
        # Learned dialogue is intentionally conservative. A novel form must have
        # clear cue evidence before it can trigger a semantic operation.
        if act is None or conf < .60:
            return None
        trace = [f"dialogue:{act}:{cue}:{weight:.3f}" for cue,weight in evidence[:4]]

        if act == "DEFINE":
            if not concepts:
                return ConversationReply("I understand that as a definition request, but I cannot reliably resolve which concept you mean.", "CLARIFY", .35, trace=trace)
            row = self._concept_definition(concepts[0]); row.confidence=min(row.confidence, conf); row.trace += trace; return row
        if act == "DEPS":
            if not concepts:
                return ConversationReply("I understand that as a dependency question, but I cannot reliably resolve the concept.", "CLARIFY", .35, trace=trace)
            row=self._dependencies(concepts[0]); row.confidence=min(row.confidence,conf); row.trace+=trace; return row
        if act == "KNOW":
            if not concepts:
                return ConversationReply("I understand that as a knowledge question, but I cannot resolve the concept.", "CLARIFY", .35, trace=trace)
            concept=concepts[0]; audit=self.concepts.understanding(concept); facts=self.facts.about(concept)
            known=bool(audit.get("known") or facts)
            text_out=f"Yes. I have explicit memory for '{concept}'." if known else f"No. I do not currently have explicit memory for '{concept}'."
            return ConversationReply(text_out,"ANSWER_KNOWLEDGE",conf,concept=concept,trace=trace)
        if act == "ABOUT":
            if not concepts:
                return ConversationReply("I understand that as a request for more information, but I cannot resolve the topic.","CLARIFY",.35,trace=trace)
            concept=concepts[0]; self.state.last_concept=concept
            first=self._concept_definition(concept)
            if first.act == "CLARIFY": first.trace+=trace; return first
            more=self._tell_more(); first.text += " " + more.text; first.confidence=min(first.confidence,conf); first.trace+=trace; return first
        if act == "COMPARE":
            if len(concepts)<2:
                return ConversationReply("I understand that as a comparison, but I need two known concepts to compare.","CLARIFY",.35,trace=trace)
            row=self._compare(concepts[0],concepts[1]); row.confidence=min(row.confidence,conf); row.trace+=trace; return row
        if act == "FOLLOW_DEPS":
            if not self.state.last_concept:
                return ConversationReply("I understand the follow-up, but there is no current concept for 'it' to refer to.","CLARIFY",.30,trace=trace)
            row=self._dependencies(self.state.last_concept); row.confidence=min(row.confidence,conf); row.trace+=trace; return row
        if act == "FOLLOW_WHY":
            row=self._why(); row.confidence=min(row.confidence,conf); row.trace+=trace; return row
        if act == "FOLLOW_MORE":
            row=self._tell_more(); row.confidence=min(row.confidence,conf); row.trace+=trace; return row
        if act == "LAST_TAUGHT":
            if self.state.last_taught:
                return ConversationReply(f"The last explicit thing I learned was about '{self.state.last_taught}'.","ANSWER_MEMORY",conf,concept=self.state.last_taught,trace=trace)
            return ConversationReply("You have not explicitly taught me anything in this dialogue yet.","ANSWER_MEMORY",conf,trace=trace)
        if act == "TOPIC":
            if self.state.last_concept:
                return ConversationReply(f"Our current semantic topic is '{self.state.last_concept}'.","ANSWER_MEMORY",conf,concept=self.state.last_concept,trace=trace)
            return ConversationReply("I do not have a current semantic topic yet.","ANSWER_MEMORY",conf,trace=trace)
        if act == "HELP":
            return ConversationReply(
                "You can ask about stored concepts, dependencies, provenance, comparisons and world memory; you can also teach aliases, executable definitions and explicit facts. If I cannot map a construction reliably, I will ask for a rephrase or demonstration.",
                "HELP",conf,trace=trace,
            )
        if act == "GREETING":
            return ConversationReply("Hello. I am ready to continue our conversation or learn from an explicit demonstration.","GREETING",conf,trace=trace)
        return None

    def _direct_construction_known(self, q: str) -> bool:
        """True when the conservative base shell already has a direct parse."""
        if self.GREETING.match(q) or self.THANKS.match(q) or self.HELP.match(q):
            return True
        if self.LAST_TAUGHT.match(q) or self.TOPIC.match(q):
            return True
        if any(p.match(q) for p in self.ALIAS_PATTERNS):
            return True
        if self.EXPLICIT_DEFINITION.match(q) or self.REMEMBER_ISA.match(q):
            return True
        if self._EXEC_DEFINITION_CUES.search(" "+q+" ") and not q.endswith("?"):
            return True
        if self.FOLLOW_DEPS.match(q) or self.FOLLOW_WHY.match(q) or self.FOLLOW_KNOW.match(q) or self.FOLLOW_MORE.match(q):
            return True
        for pattern in (self.COMPARE, self.DEPS, self.ABOUT, self.WHAT, self.KNOW, self.ISA_Q, self.WHERE):
            if pattern.match(q):
                return True
        if re.match(r"^(?:calculate|compute|evaluate|find)\b",q,re.I):
            return True
        return False

    def respond(self, text: str) -> ConversationReply:
        q=self._clean(text)

        # First preserve every already-reliable direct operation. Learned memory
        # is a generalization fallback, never a replacement for stronger evidence.
        if self._direct_construction_known(q):
            self.last_learned_dialogue_evidence = []
            return super().respond(q)

        learned=self._learned_reply(q)
        if learned is not None:
            return self._remember_reply(learned)
        return super().respond(q)

    def summary(self):
        base=super().summary()
        base["version"]=self.VERSION
        base["dialogue_learner"]=self.dialogue_learner.summary(12)
        base["last_learned_dialogue_evidence"]=list(self.last_learned_dialogue_evidence)
        return base
