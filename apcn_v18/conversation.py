from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import re

from apcn_v10.definitions import ConceptStore, normalize_name
from apcn_v15.conversation import ConversationReply
from apcn_v15.lexicon import FactMemory, LexicalSemanticMemory
from apcn_v16.bidirectional import SemanticFrame
from apcn_v17.language import StructuredLanguageV17
from apcn_v17.memory import SemanticDiscourseV17, StructuredSemanticMemoryV17
from apcn_v17.semantics import SemanticClause, normalize_term

from .realizer import AnswerPlanV18, NaturalRealizerV18


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _strip_terminal(text: str) -> str:
    return _clean(text).strip(" .?!\t\n")


def _levenshtein(a: str, b: str) -> int:
    a, b = normalize_term(a), normalize_term(b)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        nxt = [i]
        for j, cb in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + int(ca != cb)))
        row = nxt
    return row[-1]


class NaturalConversationEngineV18:
    """Unified ordinary-English -> structured-semantics conversation layer."""

    VERSION = "APCN-V0.18-NATURAL-SEMANTIC-CONVERSATION"

    TEACH = re.compile(r"^(?:remember|learn|note)\s+that\s+(.+?)[.!]*$", re.I)
    FORMAL_TRUTH = re.compile(r"^(?:is it true that|is it the case that)\s+(.+?)[?!.]*$", re.I)
    ORDINARY_COPULAR = re.compile(r"^(is|are|was|were)\s+(.+?)[?!.]*$", re.I)
    ORDINARY_EVENT = re.compile(r"^(does|do|did)\s+(.+?)[?!.]*$", re.I)
    FOLLOWS = re.compile(r"^(?:what follows if|what happens if)\s+(.+?)[?!.]*$", re.I)
    CAUSES = re.compile(r"^(?:what causes|what caused)\s+(.+?)[?!.]*$", re.I)
    BEFORE = re.compile(r"^(?:what happens before|what happened before)\s+(.+?)[?!.]*$", re.I)
    ABOUT = re.compile(r"^(?:what do you know about|tell me what you know about|what have you stored about|tell me about)\s+(.+?)[?!.]*$", re.I)
    DEFINITION = re.compile(r"^(?:(?:i\s+(?:asked|meant)\s+)(?:you\s+)?)?(?:what is|what's|define|explain)\s+(.+?)[?!.]*$", re.I)
    WHY = re.compile(r"^(?:why|why is that|how do you know|how do you know that)[?!.]*$", re.I)

    def __init__(
        self,
        structured_language: StructuredLanguageV17,
        semantic_memory: StructuredSemanticMemoryV17,
        discourse: SemanticDiscourseV17,
        realizer: NaturalRealizerV18,
        concepts: ConceptStore,
        lexicon: LexicalSemanticMemory,
        facts: FactMemory,
        *,
        legacy_language=None,
        fallback_engine=None,
    ):
        self.structured_language = structured_language
        self.semantic_memory = semantic_memory
        self.discourse = discourse
        self.realizer = realizer
        self.concepts = concepts
        self.lexicon = lexicon
        self.facts = facts
        self.legacy_language = legacy_language
        self.fallback_engine = fallback_engine
        self.turns = 0
        self.v18_turns = 0
        self.last_plan: Optional[AnswerPlanV18] = None
        self.last_reason: Optional[str] = None
        self.last_clause: Optional[SemanticClause] = None
        self.last_trace: List[str] = []

    @staticmethod
    def _walk(clause: SemanticClause):
        yield clause
        for child in clause.children:
            yield from NaturalConversationEngineV18._walk(child)

    def _render_clause(self, clause: SemanticClause) -> str:
        rows = self.structured_language.generate(clause, 1)
        if rows:
            return rows[0].rstrip(" .?!")
        return str(clause.to_dict())

    def _realize(self, plan: AnswerPlanV18, *, act: str, concept: Optional[str] = None, learned: bool = False, trace: Sequence[str] = (), clause: Optional[SemanticClause] = None, reason: Optional[str] = None) -> ConversationReply:
        self.turns += 1
        self.v18_turns += 1
        self.last_plan = plan
        self.last_reason = reason
        self.last_clause = clause
        self.last_trace = list(trace)
        if clause is not None:
            self.discourse.ingest(clause)
        return ConversationReply(self.realizer.realize(plan), act, plan.confidence, learned=learned, concept=concept, trace=["v018:natural_semantic_path", *trace])

    def _parse_statement(self, text: str) -> Tuple[Optional[SemanticClause], float, List[Dict[str, object]]]:
        surface = _strip_terminal(text)
        m = re.match(r"^(.+?)\s+(causes|cause|caused)\s+(?:that\s+)?(.+)$", surface, re.I)
        if m:
            left, lconf, lev = self.structured_language.parse(m.group(1), discourse=self.discourse)
            right, rconf, rev = self.structured_language.parse(m.group(3), discourse=self.discourse)
            if left is not None and right is not None:
                clause = SemanticClause.make("CAUSE", children=(left, right))
                evidence = [{"mode": "v018_structural_causative", "marker": m.group(2).lower()}]
                evidence.extend(lev[:1]); evidence.extend(rev[:1])
                return clause, min(.94, max(.60, min(lconf, rconf) + .10)), evidence
        return self.structured_language.parse(surface, discourse=self.discourse)

    def _known_entities(self) -> List[str]:
        out = set(self.discourse.entities)
        for rec in self.semantic_memory.records.values():
            for node in self._walk(rec.clause):
                for key in ("subject", "object"):
                    val = node.get(key)
                    if val:
                        out.add(val)
        for rec in self.facts.facts.values():
            out.add(rec.subject)
        return sorted((normalize_term(x) for x in out if normalize_term(x)), key=len, reverse=True)

    def _split_subject_predicate(self, rest: str) -> Tuple[str, str]:
        rest_n = normalize_term(rest)
        for entity in self._known_entities():
            if rest_n == entity:
                return entity, ""
            if rest_n.startswith(entity + " "):
                return entity, rest_n[len(entity):].strip()
            if rest_n.startswith("the " + entity + " "):
                return entity, rest_n[len(entity) + 5:].strip()
        bits = rest_n.split(" ", 1)
        if len(bits) == 1:
            return bits[0], ""
        subject, predicate = bits[0], bits[1]
        if subject == "the" and " " in predicate:
            bits2 = predicate.split(" ", 1)
            subject, predicate = bits2[0], bits2[1]
        return subject, predicate

    def _compile_truth_question(self, text: str) -> Optional[SemanticClause]:
        q = _clean(text)
        m = self.FORMAL_TRUTH.match(q)
        if m:
            return self._parse_statement(m.group(1))[0]
        m = self.ORDINARY_COPULAR.match(q)
        if m:
            cop, rest = m.group(1).lower(), m.group(2)
            if normalize_term(rest).startswith(("it true that ", "it the case that ")):
                return None
            subject, predicate = self._split_subject_predicate(rest)
            if not subject or not predicate:
                return None
            return self._parse_statement(f"{subject} {cop} {predicate}")[0]
        m = self.ORDINARY_EVENT.match(q)
        if m:
            aux, rest = m.group(1).lower(), m.group(2)
            subject, predicate = self._split_subject_predicate(rest)
            if not subject or not predicate:
                return None
            clause = self._parse_statement(f"{subject} {predicate}")[0]
            if clause is not None and aux == "did" and clause.op == "EVENT":
                clause = SemanticClause.make("EVENT", tense="past", **clause.slot_dict())
            return clause
        return None

    def _legacy_isa_truth(self, clause: SemanticClause) -> Optional[Dict[str, object]]:
        if clause.op != "IS_A":
            return None
        subject, category = clause.get("subject"), clause.get("category")
        if not subject or not category:
            return None
        for rec in self.facts.about(subject):
            if rec.relation == "is_a" and rec.object == category:
                return {"known": True, "truth": True, "mode": "legacy_explicit_is_a", "support": rec.support, "source": rec.source, "evidence": [clause.to_dict()]}
        return None

    def _isa_chain_inference(self, clause: SemanticClause) -> Optional[Dict[str, object]]:
        if clause.op != "IS_A":
            return None
        subject, target = clause.get("subject"), clause.get("category")
        if not subject or not target:
            return None
        direct: List[Tuple[str, SemanticClause, int]] = []
        for rec in self.semantic_memory.records.values():
            c = rec.clause
            if c.op == "IS_A" and c.get("subject") == subject and c.get("category"):
                direct.append((c.get("category") or "", c, rec.support))
        if not direct:
            for rec in self.facts.about(subject):
                if rec.relation == "is_a":
                    seed = SemanticClause.make("IS_A", subject=subject, category=rec.object, tense="present")
                    direct.append((rec.object, seed, rec.support))
        rules: Dict[str, List[Tuple[str, SemanticClause, int]]] = {}
        for rec in self.semantic_memory.records.values():
            c = rec.clause
            if c.op == "FORALL_ISA" and c.get("kind") and c.get("category"):
                rules.setdefault(c.get("kind") or "", []).append((c.get("category") or "", c, rec.support))
        queue: List[Tuple[str, List[SemanticClause], int]] = [(kind, [evidence], support) for kind, evidence, support in direct]
        seen = set()
        while queue:
            kind, evidence, support = queue.pop(0)
            if kind in seen:
                continue
            seen.add(kind)
            if kind == target:
                return {"known": True, "truth": True, "mode": "v018_transitive_universal_inference", "support": support, "source": "inference", "evidence": [c.to_dict() for c in evidence]}
            for nxt, rule, rsupport in rules.get(kind, []):
                queue.append((nxt, evidence + [rule], min(support, rsupport)))
        return None

    def _infer_truth(self, clause: SemanticClause) -> Dict[str, object]:
        result = self.semantic_memory.infer_truth(clause)
        if result.get("known"):
            return result
        chain = self._isa_chain_inference(clause)
        if chain is not None:
            return chain
        legacy = self._legacy_isa_truth(clause)
        if legacy is not None:
            return legacy
        return result

    def _evidence_reason(self, result: Dict[str, object]) -> str:
        rows: List[str] = []
        for raw in result.get("evidence", []):
            if not isinstance(raw, dict):
                continue
            try:
                clause = SemanticClause.from_dict(raw)
            except Exception:
                continue
            rows.append(self._render_clause(clause))
        if not rows:
            return "the supporting evidence is explicitly stored in my semantic memory"
        if len(rows) == 1:
            return rows[0]
        if len(rows) == 2:
            return f"{rows[0]}, and {rows[1]}"
        return ", ".join(rows[:-1]) + f", and {rows[-1]}"

    def _truth(self, clause: SemanticClause) -> ConversationReply:
        result = self._infer_truth(clause)
        statement = self._render_clause(clause)
        subject = clause.get("subject")
        mode = str(result.get("mode", "unknown"))
        if not result.get("known"):
            plan = AnswerPlanV18.make("ANSWER_UNKNOWN", statement=statement, confidence=.86)
            reason = f"I have no explicit or derivable evidence that {statement} is true"
            return self._realize(plan, act="ANSWER_UNKNOWN", concept=subject, clause=clause, reason=reason, trace=[f"v018:truth:{mode}"])
        if bool(result.get("truth")):
            reason = self._evidence_reason(result)
            plan_act = "ANSWER_TRUE_DERIVED" if "inference" in mode else "ANSWER_TRUE_EXPLICIT"
            slots = {"statement": statement}
            if plan_act == "ANSWER_TRUE_DERIVED":
                slots["reason"] = reason
            plan = AnswerPlanV18.make(plan_act, confidence=.98, **slots)
            return self._realize(plan, act="ANSWER_TRUE", concept=subject, clause=clause, reason=reason, trace=[f"v018:truth:{mode}"])
        negation = f"it is not true that {statement}"
        reason = self._evidence_reason(result)
        plan = AnswerPlanV18.make("ANSWER_FALSE", negation=negation, confidence=.98)
        return self._realize(plan, act="ANSWER_FALSE", concept=subject, clause=clause, reason=reason, trace=[f"v018:truth:{mode}"])

    def _facts_about(self, subject: str) -> Tuple[List[SemanticClause], List[SemanticClause]]:
        target = normalize_term(subject)
        explicit: List[SemanticClause] = []
        seen = set()
        for rec in self.semantic_memory.records.values():
            c = rec.clause
            if c.op in {"IS_A", "PROPERTY", "RELATION", "EVENT"} and c.get("subject") == target:
                key = str(c.to_dict())
                if key not in seen:
                    explicit.append(c); seen.add(key)
        if not explicit:
            for rec in self.facts.about(target):
                if rec.relation == "is_a":
                    c = SemanticClause.make("IS_A", subject=target, category=rec.object, tense="present")
                    key = str(c.to_dict())
                    if key not in seen:
                        explicit.append(c); seen.add(key)
        derived: List[SemanticClause] = []
        direct_categories = [c.get("category") for c in explicit if c.op == "IS_A" and c.get("category")]
        for category in direct_categories:
            for rec in self.semantic_memory.records.values():
                rule = rec.clause
                if rule.op == "FORALL_ISA" and rule.get("kind") == category and rule.get("category"):
                    c = SemanticClause.make("IS_A", subject=target, category=rule.get("category"), tense="present")
                    key = str(c.to_dict())
                    if key not in seen:
                        derived.append(c); seen.add(key)
        return explicit, derived

    def _about(self, raw_target: str) -> ConversationReply:
        target, _ = self.lexicon.resolve(raw_target)
        target = normalize_term(target)
        explicit, derived = self._facts_about(target)
        rows = [self._render_clause(c) for c in explicit]
        if derived:
            rows.extend(f"{self._render_clause(c)} (derived)" for c in derived)
        if not rows and target in self.concepts.records:
            return self._definition(target)
        if not rows:
            suggestion = self._nearest_term(target)
            plan = AnswerPlanV18.make("CLARIFY_SUGGEST", term=target, suggestion=suggestion, confidence=.72) if suggestion else AnswerPlanV18.make("CLARIFY_UNKNOWN_TERM", term=target, confidence=.70)
            return self._realize(plan, act="CLARIFY", concept=target, trace=["v018:about:unknown"])
        facts_surface = "; ".join(rows)
        plan = AnswerPlanV18.make("ANSWER_ABOUT", subject=target, facts=facts_surface, count=str(len(rows)), confidence=.96)
        reason = f"these statements are present in or derived from my explicit semantic memory: {facts_surface}"
        return self._realize(plan, act="ANSWER_ABOUT", concept=target, reason=reason, trace=["v018:about:semantic_aggregation"])

    def _definition(self, raw_target: str) -> ConversationReply:
        target = normalize_name(raw_target)
        resolved, lexical_trace = self.lexicon.resolve(target)
        resolved = normalize_name(resolved)
        rec = self.concepts.records.get(resolved)
        if rec is not None and self.legacy_language is not None:
            request = SemanticFrame.make("ASK_DEFINITION", concept=target)
            answer = self.legacy_language.responder.answer(request)
            if answer.op == "STATE_DEFINITION":
                concept = answer.get("concept") or resolved
                definition = answer.get("definition") or "an explicitly stored concept"
                plan = AnswerPlanV18.make("ANSWER_DEFINITION", concept=concept, definition=definition, confidence=.98)
                reason = f"my canonical concept memory contains {concept}" + (f" through lexical link {'; '.join(lexical_trace)}" if lexical_trace else "")
                return self._realize(plan, act="ANSWER_DEFINITION", concept=concept, reason=reason, trace=["v018:definition:canonical_concept_store", *[f"v018:lex:{x}" for x in lexical_trace]])
        explicit, derived = self._facts_about(resolved)
        if explicit or derived:
            return self._about(resolved)
        suggestion = self._nearest_term(target)
        if suggestion and suggestion != target:
            plan = AnswerPlanV18.make("CLARIFY_SUGGEST", term=target, suggestion=suggestion, confidence=.78)
            return self._realize(plan, act="CLARIFY", concept=target, trace=["v018:lexical_suggestion"])
        plan = AnswerPlanV18.make("CLARIFY_UNKNOWN_TERM", term=resolved or target, confidence=.72)
        return self._realize(plan, act="CLARIFY", concept=resolved or target, trace=["v018:definition:unknown"])

    def _nearest_term(self, term: str) -> Optional[str]:
        t = normalize_term(term)
        if not t:
            return None
        candidates = set(self.concepts.records)
        candidates.update(self.lexicon.aliases)
        candidates.update(rec.target for rec in self.lexicon.aliases.values())
        candidates.update(self.discourse.entities)
        for rec in self.semantic_memory.records.values():
            for node in self._walk(rec.clause):
                for key in ("subject", "object", "kind", "category"):
                    value = node.get(key)
                    if value:
                        candidates.add(value)
        ranked = []
        for candidate in candidates:
            c = normalize_term(candidate)
            if not c or c == t:
                continue
            d = _levenshtein(t, c)
            limit = 1 if max(len(t), len(c)) <= 5 else 2
            if d <= limit:
                ranked.append((d, abs(len(t) - len(c)), c))
        if not ranked:
            return None
        ranked.sort()
        return ranked[0][2]

    def _teach(self, surface: str) -> ConversationReply:
        clause, conf, evidence = self._parse_statement(surface)
        if clause is None or conf < .50:
            plan = AnswerPlanV18.make("CLARIFY_UNKNOWN_TERM", term="that statement", confidence=.30)
            return self._realize(plan, act="CLARIFY", trace=["v018:teaching_parse_failed"])
        self.semantic_memory.teach(clause, source="user")
        if clause.op == "IS_A" and clause.get("subject") and clause.get("category"):
            self.facts.teach(clause.get("subject") or "", "is_a", clause.get("category") or "", source="user")
        statement = self._render_clause(clause)
        plan = AnswerPlanV18.make("ACK_SEMANTIC_TEACH", statement=statement, confidence=min(.99, conf))
        return self._realize(plan, act="LEARN_STRUCTURED", concept=clause.get("subject"), learned=True, clause=clause, reason=f"you explicitly taught {statement}", trace=["v018:semantic_teaching", f"v018:op:{clause.op}", *[f"v018:evidence:{row.get('mode','unknown')}" for row in evidence[:2]]])

    def _cause(self, effect_surface: str) -> ConversationReply:
        effect = self._parse_statement(effect_surface)[0]
        if effect is None:
            plan = AnswerPlanV18.make("CLARIFY_UNKNOWN_TERM", term=effect_surface, confidence=.30)
            return self._realize(plan, act="CLARIFY", trace=["v018:cause:parse_failed"])
        rows = self.semantic_memory.causes_for(effect)
        if not rows:
            statement = self._render_clause(effect)
            plan = AnswerPlanV18.make("ANSWER_UNKNOWN", statement=statement, confidence=.80)
            return self._realize(plan, act="ANSWER_UNKNOWN", concept=effect.get("subject"), clause=effect, reason=f"I have no explicit causal rule with effect {statement}", trace=["v018:cause:none"])
        cause, relation = rows[0]["cause"], rows[0]["relation"]
        cause_s, effect_s = self._render_clause(cause), self._render_clause(effect)
        plan = AnswerPlanV18.make("ANSWER_CAUSE", cause=cause_s, effect=effect_s, confidence=.97)
        return self._realize(plan, act="ANSWER_CAUSE", concept=cause.get("subject"), clause=relation, reason=f"my semantic memory contains {self._render_clause(relation)}", trace=["v018:cause:explicit_relation"])

    def _follows(self, condition_surface: str) -> ConversationReply:
        condition = self._parse_statement(condition_surface)[0]
        if condition is None:
            plan = AnswerPlanV18.make("CLARIFY_UNKNOWN_TERM", term=condition_surface, confidence=.30)
            return self._realize(plan, act="CLARIFY", trace=["v018:conditional:parse_failed"])
        rows = self.semantic_memory.consequences_for(condition)
        if not rows:
            statement = self._render_clause(condition)
            plan = AnswerPlanV18.make("ANSWER_UNKNOWN", statement=statement, confidence=.80)
            return self._realize(plan, act="ANSWER_UNKNOWN", clause=condition, trace=["v018:conditional:none"])
        consequence, rule = rows[0]["consequence"], rows[0]["rule"]
        cond_s, cons_s = self._render_clause(condition), self._render_clause(consequence)
        plan = AnswerPlanV18.make("ANSWER_CONSEQUENCE", condition=cond_s, consequence=cons_s, confidence=.97)
        return self._realize(plan, act="ANSWER_CONSEQUENCE", concept=consequence.get("subject"), clause=rule, reason=f"my stored rule is {self._render_clause(rule)}", trace=["v018:conditional:explicit_rule"])

    def _before(self, second_surface: str) -> ConversationReply:
        second = self._parse_statement(second_surface)[0]
        if second is None:
            plan = AnswerPlanV18.make("CLARIFY_UNKNOWN_TERM", term=second_surface, confidence=.30)
            return self._realize(plan, act="CLARIFY", trace=["v018:before:parse_failed"])
        rows = self.semantic_memory.before(second)
        if not rows:
            statement = self._render_clause(second)
            plan = AnswerPlanV18.make("ANSWER_UNKNOWN", statement=statement, confidence=.80)
            return self._realize(plan, act="ANSWER_UNKNOWN", clause=second, trace=["v018:before:none"])
        first, relation = rows[0]["first"], rows[0]["relation"]
        first_s, second_s = self._render_clause(first), self._render_clause(second)
        plan = AnswerPlanV18.make("ANSWER_BEFORE", first=first_s, second=second_s, confidence=.97)
        return self._realize(plan, act="ANSWER_BEFORE", concept=first.get("subject"), clause=relation, reason=f"my temporal memory contains {self._render_clause(relation)}", trace=["v018:before:explicit_relation"])

    def _why(self) -> Optional[ConversationReply]:
        if not self.last_reason:
            return None
        plan = AnswerPlanV18.make("ANSWER_WHY", reason=self.last_reason, confidence=.95)
        return self._realize(plan, act="ANSWER_PROVENANCE", concept=None if self.last_clause is None else self.last_clause.get("subject"), clause=None, reason=self.last_reason, trace=["v018:why:last_proof"])

    def _is_legacy_explicit_teaching(self, text: str) -> bool:
        engine = getattr(self.legacy_language, "fallback_engine", None)
        if engine is None:
            return False
        q = engine._clean(text)
        if any(pattern.match(q) for pattern in engine.ALIAS_PATTERNS):
            return True
        if engine.EXPLICIT_DEFINITION.match(q):
            return True
        if engine._EXEC_DEFINITION_CUES.search(" " + q + " ") and not q.endswith("?"):
            return True
        return False

    def respond(self, text: str) -> ConversationReply:
        q = _clean(text)
        if not q:
            if self.fallback_engine is not None:
                return self.fallback_engine.respond(q)
            return ConversationReply("Say something or ask me about what I know.", "CLARIFY", .10)
        if self._is_legacy_explicit_teaching(q) and self.legacy_language is not None:
            row = self.legacy_language.respond(q)
            row.trace = list(row.trace) + ["v018:legacy_semantic_teaching"]
            self.turns += 1
            self.last_reason = None
            return row
        m = self.TEACH.match(q)
        if m:
            return self._teach(m.group(1))
        m = self.ABOUT.match(q)
        if m:
            return self._about(m.group(1))
        m = self.FOLLOWS.match(q)
        if m:
            return self._follows(m.group(1))
        m = self.CAUSES.match(q)
        if m:
            return self._cause(m.group(1))
        m = self.BEFORE.match(q)
        if m:
            return self._before(m.group(1))
        if self.WHY.match(q):
            row = self._why()
            if row is not None:
                return row
        m = self.DEFINITION.match(q)
        if m:
            return self._definition(m.group(1))
        proposition = self._compile_truth_question(q)
        if proposition is not None:
            return self._truth(proposition)
        self.turns += 1
        self.last_reason = None
        if self.fallback_engine is not None:
            row = self.fallback_engine.respond(q)
            row.trace = list(row.trace) + ["v018:fallback:v017"]
            return row
        return ConversationReply("I do not yet know how to interpret that construction reliably.", "CLARIFY", .20, trace=["v018:no_fallback"])

    def summary(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "turns": self.turns,
            "v18_turns": self.v18_turns,
            "last_plan": None if self.last_plan is None else self.last_plan.to_dict(),
            "last_reason": self.last_reason,
            "last_trace": list(self.last_trace),
            "raw_chat_transcript_persisted": False,
            "property_to_legacy_isa_bridge_enabled": False,
        }
