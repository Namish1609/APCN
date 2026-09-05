from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import re

from apcn_v10.definitions import ConceptStore, DefinitionParseError, normalize_name
from apcn_v10.query import KnowledgeQueryEngine
from .lexicon import FactMemory, LexicalSemanticMemory


@dataclass
class ConversationReply:
    text: str
    act: str
    confidence: float = 1.0
    learned: bool = False
    concept: Optional[str] = None
    trace: List[str] = field(default_factory=list)


@dataclass
class DialogueState:
    """Bounded semantic dialogue state.

    Recent state stores semantic acts/concepts rather than a permanent raw chat
    transcript. This is enough for `it`, `that`, `why?`, and follow-up questions.
    """

    last_concept: Optional[str] = None
    last_taught: Optional[str] = None
    last_act: Optional[str] = None
    pending_clarification: Optional[str] = None
    recent: List[Dict[str, object]] = field(default_factory=list)
    max_recent: int = 32

    def note(self, act: str, concept: Optional[str] = None, **meta) -> None:
        self.last_act = act
        if concept:
            self.last_concept = normalize_name(concept)
        row: Dict[str, object] = {"act": act}
        if concept:
            row["concept"] = normalize_name(concept)
        row.update(meta)
        self.recent.append(row)
        if len(self.recent) > self.max_recent:
            del self.recent[: len(self.recent) - self.max_recent]

    def semantic_summary(self) -> Dict[str, object]:
        return {
            "last_concept": self.last_concept,
            "last_taught": self.last_taught,
            "last_act": self.last_act,
            "pending_clarification": self.pending_clarification,
            "recent_semantic_turns": list(self.recent),
            "raw_chat_transcript_persisted": False,
        }


class ConversationEngine:
    """Conversational shell over explicit APCN language and knowledge memories.

    V0.15 does not call an external language model. Interpretation is a mixture
    of learned lexical links, existing APCN semantic memories, explicit dialogue
    constructions and the executable ConceptStore. Unknown language is surfaced
    as unknown instead of hallucinated.
    """

    VERSION = "APCN-V0.15-CONVERSATION-ENGINE"

    GREETING = re.compile(r"^(?:hi|hello|hey|hey there|good morning|good afternoon|good evening)[!. ]*$", re.I)
    THANKS = re.compile(r"^(?:thanks|thank you|thx|nice|great)[!. ]*$", re.I)
    HELP = re.compile(r"^(?:help|what can you do|how can i teach you|how do i teach you)[?.! ]*$", re.I)

    ALIAS_PATTERNS = (
        re.compile(r"^when i say ['\"]?(.+?)['\"]? i mean ['\"]?(.+?)['\"]?[.!]*$", re.I),
        re.compile(r"^['\"]?(.+?)['\"]? is another word for ['\"]?(.+?)['\"]?[.!]*$", re.I),
        re.compile(r"^['\"]?(.+?)['\"]? means ['\"]?(.+?)['\"]?[.!]*$", re.I),
    )
    REMEMBER_ISA = re.compile(r"^(?:remember|learn|note) that (.+?) is (?:a |an |the )?(.+?)[.!]*$", re.I)
    EXPLICIT_DEFINITION = re.compile(r"^(?:learn|remember) (?:this )?definition\s*:\s*(.+)$", re.I)

    WHAT = re.compile(r"^(?:hey\s+)?(?:what is|what's|define|explain)\s+(.+?)[?.!]*$", re.I)
    KNOW = re.compile(r"^(?:do you (?:know|understand)|are you familiar with)\s+(.+?)[?.!]*$", re.I)
    ABOUT = re.compile(r"^(?:what do you know about|tell me about)\s+(.+?)[?.!]*$", re.I)
    DEPS = re.compile(r"^(?:what does|what is)\s+(.+?)\s+(?:depend on|made from)[?.!]*$", re.I)
    ISA_Q = re.compile(r"^is\s+(.+?)\s+(?:a|an|the)\s+(.+?)[?.!]*$", re.I)
    COMPARE = re.compile(r"^(?:compare|what(?:'s| is) the difference between)\s+(.+?)\s+(?:and|with)\s+(.+?)[?.!]*$", re.I)
    WHERE = re.compile(r"^(?:where is|where's)\s+(.+?)[?.!]*$", re.I)

    FOLLOW_DEPS = re.compile(r"^(?:what does (?:it|that) depend on|what are (?:its|the) dependencies|and what does it depend on)[?.!]*$", re.I)
    FOLLOW_WHY = re.compile(r"^(?:why|why is that|how do you know|how do you know that)[?.!]*$", re.I)
    FOLLOW_KNOW = re.compile(r"^(?:do you understand (?:it|that)|do you know (?:it|that))[?.!]*$", re.I)
    FOLLOW_MORE = re.compile(r"^(?:tell me more|tell me more about (?:it|that)|and what else)[?.!]*$", re.I)
    LAST_TAUGHT = re.compile(r"^(?:what did i just teach you|what have i just taught you|what did you learn)[?.!]*$", re.I)
    TOPIC = re.compile(r"^(?:what are we talking about|what were we talking about)[?.!]*$", re.I)

    _EXEC_DEFINITION_CUES = re.compile(
        r"\b(?:divided by| per |product of|multiplied by| times |rate of change|defined as)\b", re.I
    )

    def __init__(
        self,
        concepts: ConceptStore,
        lexicon: Optional[LexicalSemanticMemory] = None,
        facts: Optional[FactMemory] = None,
        *,
        semantic_parser=None,
        discourse_registry=None,
        world_query: Optional[Callable[[str], Dict[str, object]]] = None,
    ):
        self.concepts = concepts
        self.lexicon = lexicon or LexicalSemanticMemory()
        self.facts = facts or FactMemory()
        self.state = DialogueState()
        self.semantic_parser = semantic_parser
        self.discourse_registry = discourse_registry
        self.world_query = world_query
        self.query = KnowledgeQueryEngine(concepts)
        self.turns = 0
        self.unknown_turns = 0
        self.learned_turns = 0

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", str(text).strip())

    def _resolve(self, phrase: str) -> tuple[str, List[str]]:
        return self.lexicon.resolve(phrase)

    def _remember_reply(self, reply: ConversationReply) -> ConversationReply:
        self.turns += 1
        self.unknown_turns += int(reply.act == "CLARIFY")
        self.learned_turns += int(reply.learned)
        self.state.note(reply.act, reply.concept, confidence=reply.confidence, learned=reply.learned)
        return reply

    def _concept_definition(self, name: str) -> ConversationReply:
        resolved, trace = self._resolve(name)
        audit = self.concepts.understanding(resolved)
        if audit.get("known"):
            ans = self.query.ask(f"what is {resolved}?")
            return ConversationReply(ans.answer, "ANSWER_DEFINITION", .98, concept=resolved, trace=trace)
        facts = self.facts.about(resolved)
        if facts:
            rec = facts[0]
            if rec.relation == "is_a":
                text = f"I currently know {resolved} as a {rec.object}."
            else:
                text = f"I know that {resolved} {rec.relation.replace('_',' ')} {rec.object}."
            return ConversationReply(text, "ANSWER_FACT", .93, concept=resolved, trace=trace)
        return ConversationReply(
            f"I do not currently have a grounded or explicitly taught concept for '{resolved}'. You can teach me a definition, an alias, or an explicit fact.",
            "CLARIFY", .25, concept=resolved, trace=trace,
        )

    def _dependencies(self, name: str) -> ConversationReply:
        resolved, trace = self._resolve(name)
        ans = self.query.ask(f"what does {resolved} depend on?")
        act = "ANSWER_DEPENDENCIES" if self.concepts.understanding(resolved).get("known") else "CLARIFY"
        return ConversationReply(ans.answer, act, .96 if act != "CLARIFY" else .25, concept=resolved, trace=trace)

    def _why(self) -> ConversationReply:
        concept = self.state.last_concept
        if not concept:
            return ConversationReply("I need a topic first. Ask about a concept, then ask why.", "CLARIFY", .20)
        audit = self.concepts.understanding(concept)
        if audit.get("known"):
            rec = self.concepts.records[concept]
            if rec.source_sentence:
                text = f"Because my stored concept for {concept} came from: '{rec.source_sentence}'."
                if rec.dependencies():
                    text += f" Its current direct dependencies are {', '.join(sorted(rec.dependencies()))}."
            elif rec.primitive:
                text = f"I currently treat {concept} as a {'grounded' if rec.grounded else 'partially grounded'} primitive, so I do not have a deeper concept-from-concept derivation yet."
            else:
                text = f"My current concept graph contains {concept}, but I do not have provenance detailed enough to explain it further."
            return ConversationReply(text, "ANSWER_PROVENANCE", .94, concept=concept)
        facts = self.facts.about(concept)
        if facts:
            src = facts[0].source
            return ConversationReply(f"Because that fact is in my explicit memory from {src} teaching.", "ANSWER_PROVENANCE", .90, concept=concept)
        return ConversationReply(f"I do not have enough evidence stored to explain why '{concept}' is true.", "CLARIFY", .25, concept=concept)

    def _tell_more(self) -> ConversationReply:
        concept = self.state.last_concept
        if not concept:
            return ConversationReply("Tell me which concept you want to continue with.", "CLARIFY", .20)
        audit = self.concepts.understanding(concept)
        if audit.get("known"):
            deps = list(audit.get("dependencies", []))
            unresolved = list(audit.get("unresolved", []))
            text = f"For {concept}, my stored kind is {audit.get('kind', 'unknown')} and dependency depth is {audit.get('depth', 0)}."
            if deps:
                text += f" Direct dependencies: {', '.join(deps)}."
            if unresolved:
                text += f" I still have unresolved grounding for: {', '.join(unresolved)}."
            else:
                text += " I have no unresolved downstream dependency recorded for it."
            return ConversationReply(text, "ANSWER_MORE", .94, concept=concept)
        facts = self.facts.about(concept)
        if facts:
            pieces = [f"{r.relation.replace('_',' ')} {r.object}" for r in facts[:5]]
            return ConversationReply(f"What I currently know about {concept}: " + "; ".join(pieces) + ".", "ANSWER_MORE", .90, concept=concept)
        return ConversationReply(f"I do not yet have more stored knowledge about {concept}.", "CLARIFY", .25, concept=concept)

    def _compare(self, left: str, right: str) -> ConversationReply:
        a, ta = self._resolve(left)
        b, tb = self._resolve(right)
        aa = self.concepts.understanding(a)
        bb = self.concepts.understanding(b)
        if not aa.get("known") or not bb.get("known"):
            missing = [n for n, row in ((a, aa), (b, bb)) if not row.get("known")]
            return ConversationReply(f"I cannot compare them yet because I do not know: {', '.join(missing)}.", "CLARIFY", .25, trace=ta+tb)
        da = set(aa.get("dependencies", [])); db = set(bb.get("dependencies", []))
        shared = sorted(da & db); only_a = sorted(da - db); only_b = sorted(db - da)
        text = f"{a} and {b} are both stored concepts."
        if shared:
            text += f" Shared direct dependencies: {', '.join(shared)}."
        if only_a:
            text += f" Only {a} directly uses: {', '.join(only_a)}."
        if only_b:
            text += f" Only {b} directly uses: {', '.join(only_b)}."
        if not (shared or only_a or only_b):
            text += " Their current direct dependency sets do not distinguish them."
        self.state.last_concept = a
        return ConversationReply(text, "ANSWER_COMPARE", .90, concept=a, trace=ta+tb)

    def _teach_alias(self, alias: str, target: str) -> ConversationReply:
        target_resolved, trace = self._resolve(target)
        rec = self.lexicon.teach_alias(alias, target_resolved, source="user")
        self.state.last_taught = rec.alias
        return ConversationReply(
            f"Learned: '{rec.alias}' refers to '{rec.target}'. I stored that as an explicit lexical link.",
            "LEARN_ALIAS", .99, True, rec.target, trace,
        )

    def _teach_fact(self, subject: str, obj: str) -> ConversationReply:
        s, trace_s = self._resolve(subject)
        o, trace_o = self._resolve(obj)
        rec = self.facts.teach(s, "is_a", o, source="user")
        self.state.last_taught = rec.subject
        return ConversationReply(
            f"Learned the explicit fact: {rec.subject} is a {rec.object}.",
            "LEARN_FACT", .99, True, rec.subject, trace_s + trace_o,
        )

    def _teach_definition(self, sentence: str) -> ConversationReply:
        try:
            rec = self.concepts.learn_definition(sentence)
        except DefinitionParseError as exc:
            return ConversationReply(
                f"I recognized that as teaching, but I could not map the definition into my current compositional definition language: {exc}",
                "CLARIFY", .35,
            )
        self.state.last_taught = rec.name
        return ConversationReply(
            f"Learned the concept definition for {rec.name}: {rec.definition.pretty() if rec.definition else 'stored'}.",
            "LEARN_DEFINITION", .99, True, rec.name,
        )

    def respond(self, text: str) -> ConversationReply:
        q = self._clean(text)
        if not q:
            return self._remember_reply(ConversationReply("Say something or ask me about a concept.", "CLARIFY", .10))

        if self.GREETING.match(q):
            return self._remember_reply(ConversationReply("Hello. I am ready to talk, answer from my explicit memory, or learn something you teach me.", "GREETING", .99))
        if self.THANKS.match(q):
            return self._remember_reply(ConversationReply("You're welcome.", "SOCIAL", .99))
        if self.HELP.match(q):
            return self._remember_reply(ConversationReply(
                "You can ask 'what is acceleration?', follow with 'what does it depend on?' or 'why?', teach 'fluxion means acceleration', teach an executable definition such as 'speed is distance divided by time', or say 'remember that orbix is a sensor'. I will say when I cannot interpret something.",
                "HELP", .99,
            ))

        if self.LAST_TAUGHT.match(q):
            if self.state.last_taught:
                return self._remember_reply(ConversationReply(f"The last explicit thing I learned was about '{self.state.last_taught}'.", "ANSWER_MEMORY", .98, concept=self.state.last_taught))
            return self._remember_reply(ConversationReply("You have not explicitly taught me anything in this dialogue yet.", "ANSWER_MEMORY", .98))
        if self.TOPIC.match(q):
            if self.state.last_concept:
                return self._remember_reply(ConversationReply(f"Our current semantic topic is '{self.state.last_concept}'.", "ANSWER_MEMORY", .98, concept=self.state.last_concept))
            return self._remember_reply(ConversationReply("I do not have a current semantic topic yet.", "ANSWER_MEMORY", .98))

        for pattern in self.ALIAS_PATTERNS:
            m = pattern.match(q)
            if m:
                return self._remember_reply(self._teach_alias(m.group(1), m.group(2)))

        m = self.EXPLICIT_DEFINITION.match(q)
        if m:
            return self._remember_reply(self._teach_definition(m.group(1)))

        m = self.REMEMBER_ISA.match(q)
        if m:
            return self._remember_reply(self._teach_fact(m.group(1), m.group(2)))

        # Permit concise executable definitions without requiring a special
        # command, but only when a construction cue makes teaching intent clear.
        if self._EXEC_DEFINITION_CUES.search(" " + q + " ") and not q.endswith("?"):
            return self._remember_reply(self._teach_definition(q))

        if self.FOLLOW_DEPS.match(q):
            if not self.state.last_concept:
                return self._remember_reply(ConversationReply("I need a concept to refer to before I can resolve 'it'.", "CLARIFY", .20))
            return self._remember_reply(self._dependencies(self.state.last_concept))
        if self.FOLLOW_WHY.match(q):
            return self._remember_reply(self._why())
        if self.FOLLOW_KNOW.match(q):
            if not self.state.last_concept:
                return self._remember_reply(ConversationReply("I need a current concept before I can resolve 'it'.", "CLARIFY", .20))
            concept = self.state.last_concept
            audit = self.concepts.understanding(concept)
            if audit.get("known"):
                status = "completely" if audit.get("complete") else "partially"
                return self._remember_reply(ConversationReply(f"I {status} understand '{concept}' according to my explicit grounding audit.", "ANSWER_KNOWLEDGE", .96, concept=concept))
            return self._remember_reply(ConversationReply(f"I do not currently understand '{concept}'.", "ANSWER_KNOWLEDGE", .96, concept=concept))
        if self.FOLLOW_MORE.match(q):
            return self._remember_reply(self._tell_more())

        m = self.COMPARE.match(q)
        if m:
            return self._remember_reply(self._compare(m.group(1), m.group(2)))
        m = self.DEPS.match(q)
        if m:
            return self._remember_reply(self._dependencies(m.group(1)))
        m = self.ABOUT.match(q)
        if m:
            concept, trace = self._resolve(m.group(1))
            self.state.last_concept = concept
            reply = self._concept_definition(concept)
            if reply.act != "CLARIFY":
                more = self._tell_more()
                reply.text = reply.text + " " + more.text
                reply.trace += trace
            return self._remember_reply(reply)
        m = self.WHAT.match(q)
        if m:
            return self._remember_reply(self._concept_definition(m.group(1)))
        m = self.KNOW.match(q)
        if m:
            concept, trace = self._resolve(m.group(1))
            audit = self.concepts.understanding(concept)
            facts = self.facts.about(concept)
            known = bool(audit.get("known") or facts)
            text_out = f"Yes. I have explicit memory for '{concept}'." if known else f"No. I do not currently have explicit memory for '{concept}'."
            return self._remember_reply(ConversationReply(text_out, "ANSWER_KNOWLEDGE", .97, concept=concept, trace=trace))
        m = self.ISA_Q.match(q)
        if m:
            s, ts = self._resolve(m.group(1)); o, to = self._resolve(m.group(2))
            rows = self.facts.about(s)
            yes = any(r.relation == "is_a" and r.object == o for r in rows)
            if yes:
                return self._remember_reply(ConversationReply(f"Yes. My explicit fact memory says {s} is a {o}.", "ANSWER_FACT", .98, concept=s, trace=ts+to))
            return self._remember_reply(ConversationReply(f"I do not currently have evidence in my fact memory that {s} is a {o}.", "ANSWER_FACT", .70, concept=s, trace=ts+to))
        m = self.WHERE.match(q)
        if m and self.world_query is not None:
            name, trace = self._resolve(m.group(1))
            row = self.world_query(name)
            return self._remember_reply(ConversationReply(str(row.get("answer", row)), "ANSWER_WORLD", float(row.get("confidence", .8) or .8), concept=name, trace=trace))

        # Existing executable calculation shell remains useful and is routed
        # before the generic semantic parser.
        if re.match(r"^(?:calculate|compute|evaluate|find)\b", q, re.I):
            ans = self.query.ask(q)
            act = "ANSWER_CALCULATION" if "cannot" not in ans.answer.lower() else "CLARIFY"
            return self._remember_reply(ConversationReply(ans.answer, act, .94 if act != "CLARIFY" else .30, concept=ans.concept))

        # Reuse the learned V0.14 semantic parser for grounded relational English.
        if self.semantic_parser is not None:
            try:
                program = self.semantic_parser(q, discourse_registry=self.discourse_registry)
            except TypeError:
                program = self.semantic_parser(q)
            except Exception:
                program = None
            if program is not None:
                pretty = program.pretty()
                concept = self.state.last_concept
                return self._remember_reply(ConversationReply(f"I can map that sentence into my semantic program as:\n{pretty}", "UNDERSTOOD_PROGRAM", .76, concept=concept))

        self.state.pending_clarification = q
        return self._remember_reply(ConversationReply(
            "I do not yet know how to interpret that English construction reliably. I will not invent a meaning. You can rephrase it, teach an alias/definition/fact, or use the paraphrase-teaching laboratory so I can acquire the construction.",
            "CLARIFY", .15,
        ))

    def summary(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "turns": self.turns,
            "unknown_turns": self.unknown_turns,
            "learned_turns": self.learned_turns,
            "lexicon": self.lexicon.summary(12),
            "facts": self.facts.summary(12),
            "dialogue": self.state.semantic_summary(),
        }
