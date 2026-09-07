from __future__ import annotations

from typing import Dict, List, Optional
import re

from apcn_v15.conversation import ConversationReply

from .language import StructuredLanguageV17
from .memory import SemanticDiscourseV17, StructuredSemanticMemoryV17
from .semantics import SemanticClause


class StructuredConversationEngineV17:
    """Conversation layer for explicit higher-order semantics.

    V0.17 handles explicit teaching and a small transparent reasoning surface for
    negation, cause, conditionals, temporal order, quantifiers and reference.
    Unsupported language falls back to V0.16 rather than being guessed.
    """

    VERSION = "APCN-V0.17-STRUCTURED-CONVERSATION"

    TEACH = re.compile(r"^(?:remember|learn|note)\s+that\s+(.+?)[.!]*$", re.I)
    TRUTH = re.compile(r"^(?:is it true that|is it the case that)\s+(.+?)[?!.]*$", re.I)
    FOLLOWS = re.compile(r"^(?:what follows if|what happens if)\s+(.+?)[?!.]*$", re.I)
    CAUSES = re.compile(r"^(?:what causes|what caused)\s+(.+?)[?!.]*$", re.I)
    BEFORE = re.compile(r"^(?:what happens before|what happened before)\s+(.+?)[?!.]*$", re.I)
    WHY = re.compile(r"^(?:why|why is that)[?!.]*$", re.I)

    def __init__(
        self,
        language: StructuredLanguageV17,
        memory: StructuredSemanticMemoryV17,
        discourse: SemanticDiscourseV17,
        *,
        fallback_engine=None,
    ):
        self.language = language
        self.memory = memory
        self.discourse = discourse
        self.fallback_engine = fallback_engine
        self.turns = 0
        self.structured_turns = 0
        self.last_structured_clause: Optional[SemanticClause] = None
        self.last_operation: Optional[str] = None

    def _render(self, clause: SemanticClause) -> str:
        rows = self.language.generate(clause, 1)
        return rows[0] if rows else str(clause.to_dict())

    def _parse(self, text: str) -> tuple[Optional[SemanticClause], float, List[Dict[str, object]]]:
        return self.language.parse(text, discourse=self.discourse)

    def _note(self, clause: Optional[SemanticClause], operation: str) -> None:
        self.turns += 1
        self.structured_turns += 1
        self.last_operation = operation
        if clause is not None:
            self.last_structured_clause = clause
            self.discourse.ingest(clause)

    def _teach(self, surface: str, original_text: str) -> ConversationReply:
        clause, conf, evidence = self._parse(surface)
        if clause is None or conf < .50:
            self._note(None, "CLARIFY")
            return ConversationReply(
                "I recognized explicit teaching, but I could not compile the statement into a reliable semantic structure. Please rephrase it or teach the construction explicitly.",
                "CLARIFY", .30, trace=["v017:teaching_parse_failed"],
            )
        rec = self.memory.teach(clause, source="user")
        # Preserve V0.15/V0.16 simple fact learning when their existing teaching
        # shell can safely consume the same statement. This is a bridge between
        # old concept/fact memory and the new structured proposition memory.
        if clause.op in {"IS_A", "PROPERTY"} and self.fallback_engine is not None:
            try:
                self.fallback_engine.respond(original_text)
            except Exception:
                pass
        self._note(clause, "LEARN_STRUCTURED")
        trace = ["v017:semantic_teaching", f"v017:op:{clause.op}"]
        trace.extend(f"v017:evidence:{row.get('mode','unknown')}" for row in evidence[:3])
        return ConversationReply(
            f"Stored as explicit semantic memory: {self._render(clause)}",
            "LEARN_STRUCTURED", min(.99, conf), learned=True,
            concept=clause.get("subject") or clause.get("kind"), trace=trace,
        )

    def _truth(self, surface: str) -> ConversationReply:
        clause, conf, evidence = self._parse(surface)
        if clause is None:
            self._note(None, "CLARIFY")
            return ConversationReply("I cannot reliably compile that proposition yet.", "CLARIFY", .25, trace=["v017:truth_parse_failed"])
        result = self.memory.infer_truth(clause)
        self._note(clause, "ANSWER_TRUTH")
        trace = [f"v017:truth:{result['mode']}"] + [f"v017:evidence:{row.get('mode','unknown')}" for row in evidence[:2]]
        if not result["known"]:
            return ConversationReply(
                f"I do not have explicit or derivable evidence that {self._render(clause).rstrip('.')} is true.",
                "ANSWER_UNKNOWN", min(.80, conf), concept=clause.get("subject"), trace=trace,
            )
        if result["truth"]:
            return ConversationReply(
                f"Yes. {self._render(clause)}",
                "ANSWER_TRUE", min(.97, conf), concept=clause.get("subject"), trace=trace,
            )
        return ConversationReply(
            f"No. My explicit memory supports the negation of: {self._render(clause)}",
            "ANSWER_FALSE", min(.97, conf), concept=clause.get("subject"), trace=trace,
        )

    def _follows(self, surface: str) -> ConversationReply:
        condition, conf, evidence = self._parse(surface)
        if condition is None:
            self._note(None, "CLARIFY")
            return ConversationReply("I cannot resolve the condition reliably.", "CLARIFY", .25)
        rows = self.memory.consequences_for(condition)
        self._note(condition, "ANSWER_CONSEQUENCE")
        if not rows:
            return ConversationReply(
                f"I have no explicit conditional rule whose condition is {self._render(condition).rstrip('.')}.",
                "ANSWER_UNKNOWN", min(.75, conf), trace=["v017:conditional:none"],
            )
        consequence = rows[0]["consequence"]
        rule = rows[0]["rule"]
        return ConversationReply(
            f"According to my stored conditional: {self._render(rule)} Therefore the consequence is: {self._render(consequence)}",
            "ANSWER_CONSEQUENCE", min(.96, conf),
            concept=consequence.get("subject"), trace=["v017:conditional:explicit_rule"],
        )

    def _causes(self, surface: str) -> ConversationReply:
        effect, conf, _ = self._parse(surface)
        if effect is None:
            self._note(None, "CLARIFY")
            return ConversationReply("I cannot resolve the effect reliably.", "CLARIFY", .25)
        rows = self.memory.causes_for(effect)
        self._note(effect, "ANSWER_CAUSE")
        if not rows:
            return ConversationReply(
                f"I do not have an explicit cause stored for {self._render(effect).rstrip('.')}.",
                "ANSWER_UNKNOWN", min(.75, conf), trace=["v017:cause:none"],
            )
        cause = rows[0]["cause"]
        relation = rows[0]["relation"]
        return ConversationReply(
            f"My stored causal relation is: {self._render(relation)} The cause is: {self._render(cause)}",
            "ANSWER_CAUSE", min(.96, conf), concept=cause.get("subject"), trace=["v017:cause:explicit_relation"],
        )

    def _before(self, surface: str) -> ConversationReply:
        second, conf, _ = self._parse(surface)
        if second is None:
            self._note(None, "CLARIFY")
            return ConversationReply("I cannot resolve the temporal event reliably.", "CLARIFY", .25)
        rows = self.memory.before(second)
        self._note(second, "ANSWER_BEFORE")
        if not rows:
            return ConversationReply(
                f"I do not have an explicit earlier event stored for {self._render(second).rstrip('.')}.",
                "ANSWER_UNKNOWN", min(.75, conf), trace=["v017:before:none"],
            )
        first = rows[0]["first"]
        relation = rows[0]["relation"]
        return ConversationReply(
            f"My stored temporal relation is: {self._render(relation)} The earlier event is: {self._render(first)}",
            "ANSWER_BEFORE", min(.96, conf), concept=first.get("subject"), trace=["v017:before:explicit_relation"],
        )

    def _why(self) -> Optional[ConversationReply]:
        if self.last_structured_clause is None:
            return None
        rows = self.memory.causes_for(self.last_structured_clause)
        if not rows:
            return None
        cause = rows[0]["cause"]
        relation = rows[0]["relation"]
        self._note(self.last_structured_clause, "ANSWER_CAUSE")
        return ConversationReply(
            f"Because my explicit memory contains: {self._render(relation)}",
            "ANSWER_CAUSE", .94, concept=cause.get("subject"), trace=["v017:why:causal_memory"],
        )

    def respond(self, text: str) -> ConversationReply:
        q = re.sub(r"\s+", " ", str(text).strip())
        m = self.TEACH.match(q)
        if m:
            return self._teach(m.group(1), q)
        m = self.TRUTH.match(q)
        if m:
            return self._truth(m.group(1))
        m = self.FOLLOWS.match(q)
        if m:
            return self._follows(m.group(1))
        m = self.CAUSES.match(q)
        if m:
            return self._causes(m.group(1))
        m = self.BEFORE.match(q)
        if m:
            return self._before(m.group(1))
        if self.WHY.match(q):
            reply = self._why()
            if reply is not None:
                return reply
        self.turns += 1
        if self.fallback_engine is not None:
            row = self.fallback_engine.respond(q)
            row.trace = list(row.trace) + ["v017:fallback:v016"]
            return row
        return ConversationReply("I do not know how to interpret that construction yet.", "CLARIFY", .20, trace=["v017:no_fallback"])

    def summary(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "turns": self.turns,
            "structured_turns": self.structured_turns,
            "last_operation": self.last_operation,
            "last_structured_clause": None if self.last_structured_clause is None else self.last_structured_clause.to_dict(),
            "raw_chat_transcript_persisted": False,
        }
