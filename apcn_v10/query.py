from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import re

from .definitions import ConceptStore, normalize_name


@dataclass
class QueryAnswer:
    question: str
    concept: Optional[str]
    answer: str
    trace_concept: Optional[str] = None


class KnowledgeQueryEngine:
    """Small natural-language shell over APCN's explicit concept memory.

    This is intentionally not a generative LLM. Answers are produced only from
    ConceptStore records and executable definition graphs, so unknown concepts
    stay explicitly unknown rather than being hallucinated.
    """

    WHAT = re.compile(r"^(?:hey\s+)?(?:what\s+is|what's|define|explain)\s+(.+?)[?.!]*$", re.I)
    KNOW = re.compile(r"^(?:do\s+you\s+(?:know|understand)|are\s+you\s+familiar\s+with)\s+(.+?)[?.!]*$", re.I)
    DEPS = re.compile(r"^(?:what\s+does|what\s+is)\s+(.+?)\s+(?:depend\s+on|made\s+from)[?.!]*$", re.I)
    CALC = re.compile(r"^(?:calculate|compute|evaluate|find)\s+(.+?)(?:\s+if|\s+when|\s+with)\s+(.+?)[?.!]*$", re.I)
    ASSIGN = re.compile(r"([a-zA-Z][a-zA-Z _-]*?)\s*(?:=|is)\s*(-?\d+(?:\.\d+)?)")

    def __init__(self, store: ConceptStore):
        self.store = store

    def ask(self, question: str) -> QueryAnswer:
        q = re.sub(r"\s+", " ", question.strip())
        if not q:
            return QueryAnswer(question, None, "Ask about a learned concept, for example: what is acceleration?")

        m = self.CALC.match(q)
        if m:
            return self._calculate(q, m.group(1), m.group(2))
        m = self.DEPS.match(q)
        if m:
            name = normalize_name(m.group(1))
            return self._dependencies(q, name)
        m = self.KNOW.match(q)
        if m:
            name = normalize_name(m.group(1))
            audit = self.store.understanding(name)
            if not audit.get("known"):
                return QueryAnswer(q, name, f"I do not currently have a concept for '{name}'.", name)
            if audit.get("complete"):
                return QueryAnswer(q, name, f"Yes. '{name}' is present and its current dependency graph is complete.", name)
            unresolved = ", ".join(audit.get("unresolved", [])) or "unknown dependencies"
            return QueryAnswer(q, name, f"I have a partial concept for '{name}', but its grounding is incomplete. Unresolved: {unresolved}.", name)
        m = self.WHAT.match(q)
        if m:
            return self._define(q, normalize_name(m.group(1)))

        name = normalize_name(q)
        if name in self.store.records:
            return self._define(q, name)
        return QueryAnswer(q, None,
            "I could not map that question to my current concept-memory query operations. "
            "Try 'what is acceleration?', 'what does force depend on?', or "
            "'calculate acceleration if velocity change = 20 and time = 4'.")

    def _define(self, question: str, name: str) -> QueryAnswer:
        audit = self.store.understanding(name)
        if not audit.get("known"):
            return QueryAnswer(question, name, f"I do not currently know the concept '{name}'.", name)
        rec = self.store.records[name]
        if rec.primitive:
            grounding = "grounded primitive" if rec.grounded else "primitive with incomplete grounding"
            return QueryAnswer(question, name,
                f"{name}: {grounding}. It has no concept-from-concept definition in memory yet.", name)
        definition = rec.definition.pretty() if rec.definition is not None else "no executable definition"
        deps = ", ".join(sorted(rec.dependencies())) or "none"
        if audit.get("complete"):
            status = "dependency graph complete"
        else:
            unresolved = ", ".join(audit.get("unresolved", [])) or "unknown"
            status = f"incomplete grounding; unresolved: {unresolved}"
        source = f" Learned from: \"{rec.source_sentence}\"." if rec.source_sentence else ""
        return QueryAnswer(question, name,
            f"{name} = {definition}. Dependencies: {deps}. Status: {status}.{source}", name)

    def _dependencies(self, question: str, name: str) -> QueryAnswer:
        audit = self.store.understanding(name)
        if not audit.get("known"):
            return QueryAnswer(question, name, f"I do not currently know '{name}'.", name)
        deps = audit.get("dependencies", [])
        unresolved = audit.get("unresolved", [])
        text = f"'{name}' directly depends on: {', '.join(deps) if deps else 'no other stored concepts'}."
        if unresolved:
            text += f" Unresolved downstream dependencies: {', '.join(unresolved)}."
        return QueryAnswer(question, name, text, name)

    def _calculate(self, question: str, concept_text: str, values_text: str) -> QueryAnswer:
        name = normalize_name(concept_text)
        values: Dict[str, float] = {}
        for key, raw in self.ASSIGN.findall(values_text):
            values[normalize_name(key)] = float(raw)
        if not values:
            return QueryAnswer(question, name,
                "I recognized a calculation request but no numeric assignments. Use e.g. "
                "'calculate acceleration if velocity change = 20 and time = 4'.", name)
        try:
            result = self.store.evaluate(name, values)
        except Exception as exc:
            return QueryAnswer(question, name,
                f"I cannot execute '{name}' with those values: {exc}", name)
        inputs = ", ".join(f"{k}={v:g}" for k, v in values.items())
        return QueryAnswer(question, name, f"Using my stored definition: {name} = {result:g} ({inputs}).", name)
