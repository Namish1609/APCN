from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import math
import re

from .memory import SemanticDiscourseV17
from .semantics import SemanticClause, normalize_term


_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)
_SLOT = re.compile(r"\{([a-z_]+)\}")


def _norm_surface(text: str) -> str:
    return " ".join(_TOKEN.findall(str(text).lower())).strip()


def _strip_article(text: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", normalize_term(text), flags=re.I)


def _lemma_word(word: str) -> str:
    word = normalize_term(word)
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ied"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ed"):
        base = word[:-2]
        if len(base) >= 2 and base[-1] == base[-2] and base[-1] not in "aeiou":
            base = base[:-1]
        elif base.endswith("i"):
            base = base[:-1] + "y"
        elif base and not base.endswith("e") and word[:-1].endswith("e"):
            base = word[:-1]
        return base
    if len(word) > 3 and word.endswith("es") and word[-3:] in {"ses", "xes", "zes"}:
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _lemma_event(text: str) -> str:
    parts = normalize_term(text).split()
    if not parts:
        return ""
    if parts[0] == "will" and len(parts) > 1:
        parts = parts[1:]
    parts[0] = _lemma_word(parts[0])
    return " ".join(parts)


def _present_event(base: str) -> str:
    parts = normalize_term(base).split()
    if not parts:
        return base
    first = parts[0]
    if first.endswith("y") and len(first) > 1 and first[-2] not in "aeiou":
        first = first[:-1] + "ies"
    elif first.endswith(("s", "x", "z", "ch", "sh")):
        first = first + "es"
    else:
        first = first + "s"
    return " ".join([first] + parts[1:])


def _past_event(base: str) -> str:
    parts = normalize_term(base).split()
    if not parts:
        return base
    first = parts[0]
    if first.endswith("e"):
        first = first + "d"
    elif first.endswith("y") and len(first) > 1 and first[-2] not in "aeiou":
        first = first[:-1] + "ied"
    elif len(first) >= 3 and first[-1] not in "aeiouwxy" and first[-2] in "aeiou" and first[-3] not in "aeiou":
        first = first + first[-1] + "ed"
    else:
        first = first + "ed"
    return " ".join([first] + parts[1:])


class AtomicGrammarV17:
    """Small structural atom parser/realizer beneath V0.17 operators.

    This layer handles entity/property/category/relation/event shape. It does not
    decide whether an atomic proposition is true; truth remains in explicit
    semantic/world memory.
    """

    RELATIONS = ("left of", "right of", "inside", "within", "near", "above", "below", "under", "over")

    @staticmethod
    def _resolve(term: str, discourse: Optional[SemanticDiscourseV17]) -> str:
        term_n = _strip_article(term)
        return discourse.resolve(term_n) if discourse is not None else term_n

    @classmethod
    def _subject_split(cls, text: str, discourse: Optional[SemanticDiscourseV17]) -> Tuple[str, str]:
        text = _norm_surface(text)
        if discourse is not None:
            known = sorted(discourse.entities, key=len, reverse=True)
            for entity in known:
                if text == entity:
                    return entity, ""
                if text.startswith(entity + " "):
                    return entity, text[len(entity):].strip()
        parts = text.split(" ", 1)
        if len(parts) == 1:
            return cls._resolve(parts[0], discourse), ""
        return cls._resolve(parts[0], discourse), parts[1]

    @classmethod
    def parse(cls, text: str, discourse: Optional[SemanticDiscourseV17] = None) -> Optional[SemanticClause]:
        s = _norm_surface(text)
        if not s:
            return None

        # Inline copular negation is structural sugar over NOT(...).
        m = re.match(r"^(.+?)\s+(?:is|are|was|were)\s+not\s+(.+)$", s)
        if m:
            positive = cls.parse(f"{m.group(1)} is {m.group(2)}", discourse)
            return None if positive is None else SemanticClause.make("NOT", children=(positive,))

        for rel in cls.RELATIONS:
            m = re.match(rf"^(.+?)\s+(?:is|are|was|were)\s+{re.escape(rel)}\s+(.+)$", s)
            if m:
                tense = "past" if re.search(r"\b(?:was|were)\b", s) else "present"
                return SemanticClause.make(
                    "RELATION", tense=tense,
                    subject=cls._resolve(m.group(1), discourse), relation=rel,
                    object=cls._resolve(m.group(2), discourse),
                )

        m = re.match(r"^(.+?)\s+(is|are|was|were)\s+(?:a|an)\s+(.+)$", s)
        if m:
            tense = "past" if m.group(2) in {"was", "were"} else "present"
            return SemanticClause.make(
                "IS_A", tense=tense,
                subject=cls._resolve(m.group(1), discourse), category=_strip_article(m.group(3)),
            )

        m = re.match(r"^(.+?)\s+(is|are|was|were)\s+(.+)$", s)
        if m:
            tense = "past" if m.group(2) in {"was", "were"} else "present"
            return SemanticClause.make(
                "PROPERTY", tense=tense,
                subject=cls._resolve(m.group(1), discourse), property=normalize_term(m.group(3)),
            )

        subject, remainder = cls._subject_split(s, discourse)
        if not remainder:
            return SemanticClause.make("ATOM", text=s, subject=subject)
        if remainder.startswith("will "):
            return SemanticClause.make("EVENT", tense="future", subject=subject, event=_lemma_event(remainder[5:]))

        parts = remainder.split()
        if not parts:
            return None
        event_tense = "past" if parts[0].endswith("ed") else "present"
        # Common phrasal-particle form is treated as one event, e.g. turn off.
        if len(parts) == 2 and parts[1] in {"on", "off", "up", "down", "out", "in"}:
            return SemanticClause.make("EVENT", tense=event_tense, subject=subject, event=_lemma_event(remainder))
        if len(parts) == 1:
            return SemanticClause.make("EVENT", tense=event_tense, subject=subject, event=_lemma_event(remainder))
        # Generic transitive surface: first lexical verb + remaining object phrase.
        return SemanticClause.make(
            "EVENT", tense=event_tense, subject=subject,
            event=_lemma_word(parts[0]), object=cls._resolve(" ".join(parts[1:]), discourse),
        )

    @classmethod
    def generate(cls, clause: SemanticClause) -> str:
        slots = clause.slot_dict()
        if clause.op == "IS_A":
            cop = "was" if clause.tense == "past" else "is"
            return f"{slots['subject']} {cop} a {slots['category']}"
        if clause.op == "PROPERTY":
            cop = "was" if clause.tense == "past" else "is"
            return f"{slots['subject']} {cop} {slots['property']}"
        if clause.op == "RELATION":
            cop = "was" if clause.tense == "past" else "is"
            return f"{slots['subject']} {cop} {slots['relation']} {slots['object']}"
        if clause.op == "EVENT":
            event = slots.get("event", "")
            if clause.tense == "future":
                pred = "will " + event
            elif clause.tense == "past":
                pred = _past_event(event)
            else:
                pred = _present_event(event)
            if slots.get("object"):
                pred += " " + slots["object"]
            return f"{slots['subject']} {pred}"
        if clause.op == "ATOM":
            return slots.get("text", "")
        raise ValueError(f"not an atomic clause: {clause.op}")


@dataclass
class OperatorConstruction:
    op: str
    template: str
    roles: Tuple[str, ...]
    support: int = 0
    parse_hits: int = 0
    generation_hits: int = 0

    def key(self) -> str:
        return f"{self.op}|{self.template}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "op": self.op,
            "template": self.template,
            "roles": list(self.roles),
            "support": self.support,
            "parse_hits": self.parse_hits,
            "generation_hits": self.generation_hits,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "OperatorConstruction":
        return cls(
            str(data["op"]), str(data["template"]), tuple(str(x) for x in data.get("roles", [])),
            int(data.get("support", 0)), int(data.get("parse_hits", 0)), int(data.get("generation_hits", 0)),
        )


class OperatorConstructionMemoryV17:
    """Bounded learned semantic-operator surface memory used both ways."""

    VERSION = "APCN-V0.17-OPERATOR-CONSTRUCTIONS"

    def __init__(self, max_patterns: int = 1024):
        self.max_patterns = int(max_patterns)
        self.records: Dict[str, OperatorConstruction] = {}
        self.observations = 0
        self.raw_sentences_retained = 0

    def observe(self, op: str, template: str, weight: int = 1) -> OperatorConstruction:
        template_n = _norm_surface(template.replace("{", "zzslotopenzz").replace("}", "zzslotclosezz"))
        template_n = template_n.replace("zzslotopenzz", "{").replace("zzslotclosezz", "}")
        roles = tuple(m.group(1) for m in _SLOT.finditer(template_n))
        rec = OperatorConstruction(str(op).upper(), template_n, tuple(dict.fromkeys(roles)))
        key = rec.key()
        old = self.records.get(key)
        if old is None:
            self.records[key] = rec
            old = rec
        old.support += max(1, int(weight))
        self.observations += max(1, int(weight))
        self._prune()
        return old

    def _prune(self) -> None:
        if len(self.records) <= self.max_patterns:
            return
        ranked = sorted(self.records.values(), key=lambda r: (r.support + r.parse_hits + r.generation_hits, r.support, r.key()))
        for rec in ranked[: len(self.records) - self.max_patterns]:
            self.records.pop(rec.key(), None)

    @staticmethod
    def regex_for(template: str) -> re.Pattern[str]:
        parts: List[str] = []
        pos = 0
        seen = set()
        for m in _SLOT.finditer(template):
            literal = re.escape(template[pos:m.start()]).replace(r"\ ", r"\s+")
            parts.append(literal)
            role = m.group(1)
            if role in seen:
                parts.append(rf"(?P={role})")
            else:
                parts.append(rf"(?P<{role}>.+?)")
                seen.add(role)
            pos = m.end()
        parts.append(re.escape(template[pos:]).replace(r"\ ", r"\s+"))
        return re.compile(r"^" + "".join(parts) + r"$", re.I)

    def summary(self, limit: int = 20) -> Dict[str, object]:
        by_op = Counter(rec.op for rec in self.records.values())
        rows = sorted(self.records.values(), key=lambda r: (r.support, r.parse_hits + r.generation_hits, r.template), reverse=True)[:limit]
        return {
            "version": self.VERSION,
            "observations": self.observations,
            "patterns": len(self.records),
            "max_patterns": self.max_patterns,
            "by_op": dict(by_op),
            "strongest": [r.to_dict() for r in rows],
            "raw_sentences_retained": 0,
            "world_facts_retained": 0,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "max_patterns": self.max_patterns,
            "observations": self.observations,
            "records": [r.to_dict() for r in self.records.values()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "OperatorConstructionMemoryV17":
        obj = cls(int(data.get("max_patterns", 1024)))
        obj.observations = int(data.get("observations", 0))
        for row in data.get("records", []):
            rec = OperatorConstruction.from_dict(row)  # type: ignore[arg-type]
            obj.records[rec.key()] = rec
        obj._prune()
        return obj

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "OperatorConstructionMemoryV17":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class StructuredTeacherV17:
    """Paired semantic-operator surface curriculum.

    TRAIN is used for bootstrap. RECOMBINATION is frozen diagnostic material and
    is not observed by bootstrap.
    """

    TRAIN: Dict[str, Tuple[str, ...]] = {
        "NOT": (
            "not {clause}",
            "it is false that {clause}",
            "it is not true that {clause}",
        ),
        "CAUSE": (
            "{effect} because {cause}",
            "{cause} causes that {effect}",
            "because {cause} therefore {effect}",
        ),
        "IF": (
            "if {condition} then {consequence}",
            "{consequence} if {condition}",
            "provided {condition} then {consequence}",
        ),
        "BEFORE": (
            "{first} before {second}",
            "before {second} comes {first}",
        ),
        "AFTER": (
            "{second} after {first}",
            "after {first} comes {second}",
        ),
        "AND": (
            "both {left} and {right}",
            "{left} and also {right}",
        ),
        "FORALL_ISA": (
            "every {kind} is a {category}",
            "all {kind} are {category}",
            "each {kind} is a {category}",
        ),
        "EXISTS_PROPERTY": (
            "some {kind} is {property}",
            "there is a {kind} that is {property}",
            "at least one {kind} is {property}",
        ),
    }

    RECOMBINATION: Dict[str, Tuple[str, ...]] = {
        "CAUSE": ("if {cause} is the reason then {effect}",),
        "IF": ("then {consequence} whenever {condition}",),
        "BEFORE": ("{first} happens earlier than {second}",),
        "FORALL_ISA": ("any {kind} is a {category}",),
    }

    def bootstrap(self, memory: OperatorConstructionMemoryV17, repeats: int = 4) -> Dict[str, object]:
        before = memory.observations
        for op, templates in self.TRAIN.items():
            for template in templates:
                memory.observe(op, template, repeats)
        return {
            "observations_added": memory.observations - before,
            "patterns": len(memory.records),
            "recombination_templates_used": 0,
        }


class StructuredLanguageV17:
    """Bidirectional nested semantics over learned operator constructions."""

    VERSION = "APCN-V0.17-STRUCTURED-BIDIRECTIONAL-LANGUAGE"
    NESTED_OPS = {"NOT", "CAUSE", "IF", "BEFORE", "AFTER", "AND"}

    def __init__(self, memory: OperatorConstructionMemoryV17):
        self.memory = memory
        self.last_evidence: List[Dict[str, object]] = []

    @staticmethod
    def _literal_weight(template: str) -> float:
        literal = _SLOT.sub(" ", template)
        return 1.0 + math.log1p(len(_TOKEN.findall(literal)))

    def _from_match(
        self,
        rec: OperatorConstruction,
        match: re.Match[str],
        discourse: Optional[SemanticDiscourseV17],
        depth: int,
    ) -> Optional[SemanticClause]:
        vals = {role: match.group(role).strip() for role in rec.roles if match.groupdict().get(role)}
        if rec.op == "FORALL_ISA":
            if not vals.get("kind") or not vals.get("category"):
                return None
            return SemanticClause.make("FORALL_ISA", kind=_strip_article(vals["kind"]), category=_strip_article(vals["category"]))
        if rec.op == "EXISTS_PROPERTY":
            if not vals.get("kind") or not vals.get("property"):
                return None
            return SemanticClause.make("EXISTS_PROPERTY", kind=_strip_article(vals["kind"]), property=normalize_term(vals["property"]))
        role_orders = {
            "NOT": ("clause",),
            "CAUSE": ("cause", "effect"),
            "IF": ("condition", "consequence"),
            "BEFORE": ("first", "second"),
            "AFTER": ("first", "second"),
            "AND": ("left", "right"),
        }
        order = role_orders.get(rec.op)
        if order is None:
            return None
        children = []
        for role in order:
            child = self.parse(vals.get(role, ""), discourse=discourse, depth=depth + 1)[0]
            if child is None:
                return None
            children.append(child)
        return SemanticClause.make(rec.op, children=tuple(children))

    def parse(
        self,
        text: str,
        *,
        discourse: Optional[SemanticDiscourseV17] = None,
        depth: int = 0,
    ) -> Tuple[Optional[SemanticClause], float, List[Dict[str, object]]]:
        if depth > 8:
            return None, 0.0, []
        surface = _norm_surface(text)
        candidates: List[Tuple[float, OperatorConstruction, SemanticClause]] = []
        for rec in self.memory.records.values():
            match = self.memory.regex_for(rec.template).match(surface)
            if not match:
                continue
            clause = self._from_match(rec, match, discourse, depth)
            if clause is None:
                continue
            score = math.log1p(rec.support) * self._literal_weight(rec.template) + .15 * clause.structural_size()
            candidates.append((score, rec, clause))
        if candidates:
            candidates.sort(key=lambda row: (row[0], row[1].support, row[1].template), reverse=True)
            top_score, top_rec, top_clause = candidates[0]
            second = candidates[1][0] if len(candidates) > 1 else 0.0
            margin = top_score / max(top_score + second, 1e-9)
            support_factor = 1.0 - math.exp(-top_rec.support / 3.0)
            confidence = min(.99, .55 * margin + .45 * support_factor)
            top_rec.parse_hits += 1
            evidence = [
                {"mode": "operator_construction", "op": rec.op, "template": rec.template, "support": rec.support, "score": float(score)}
                for score, rec, _ in candidates[:5]
            ]
            self.last_evidence = evidence
            return top_clause, float(confidence), evidence

        atom = AtomicGrammarV17.parse(surface, discourse)
        if atom is None:
            self.last_evidence = []
            return None, 0.0, []
        evidence = [{"mode": "atomic_grammar", "op": atom.op}]
        self.last_evidence = evidence
        return atom, .78, evidence

    @staticmethod
    def _finish(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return text
        return text[0].upper() + text[1:].rstrip(".?!") + "."

    def generate(self, clause: SemanticClause, limit: int = 12) -> List[str]:
        if clause.op in {"ATOM", "IS_A", "PROPERTY", "RELATION", "EVENT"}:
            return [self._finish(AtomicGrammarV17.generate(clause))]

        rows: List[Tuple[int, str, OperatorConstruction]] = []
        for rec in self.memory.records.values():
            if rec.op != clause.op:
                continue
            role_values: Dict[str, str] = {}
            if clause.op == "FORALL_ISA":
                role_values = {"kind": clause.get("kind", "") or "", "category": clause.get("category", "") or ""}
            elif clause.op == "EXISTS_PROPERTY":
                role_values = {"kind": clause.get("kind", "") or "", "property": clause.get("property", "") or ""}
            else:
                role_orders = {
                    "NOT": ("clause",),
                    "CAUSE": ("cause", "effect"),
                    "IF": ("condition", "consequence"),
                    "BEFORE": ("first", "second"),
                    "AFTER": ("first", "second"),
                    "AND": ("left", "right"),
                }
                roles = role_orders.get(clause.op)
                if roles is None or len(roles) != len(clause.children):
                    continue
                for role, child in zip(roles, clause.children):
                    child_rows = self.generate(child, 1)
                    if not child_rows:
                        role_values = {}
                        break
                    role_values[role] = child_rows[0].rstrip(".?!")
            if any(not role_values.get(role) for role in rec.roles):
                continue
            surface = rec.template
            for role in rec.roles:
                surface = surface.replace("{" + role + "}", role_values[role])
            if _SLOT.search(surface):
                continue
            rows.append((rec.support, self._finish(surface), rec))

        rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
        out: List[str] = []
        seen = set()
        for _, text, rec in rows:
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
            rec.generation_hits += 1
            if len(out) >= max(1, int(limit)):
                break
        return out

    def roundtrip(self, clause: SemanticClause, limit: int = 12) -> Dict[str, object]:
        surfaces = self.generate(clause, limit)
        rows = []
        exact = 0
        for surface in surfaces:
            parsed, conf, evidence = self.parse(surface)
            ok = parsed == clause
            exact += int(ok)
            rows.append({
                "surface": surface,
                "parsed": None if parsed is None else parsed.to_dict(),
                "confidence": conf,
                "exact": ok,
                "evidence": evidence[:2],
            })
        return {
            "semantic": clause.to_dict(),
            "count": len(rows),
            "roundtrip_exact": exact / max(1, len(rows)),
            "generations": rows,
        }

    def summary(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "operator_memory": self.memory.summary(16),
            "last_evidence": list(self.last_evidence),
            "surface_generator_has_truth_memory_reference": False,
        }
