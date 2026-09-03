from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import json
import math
import re


@dataclass(frozen=True)
class Expr:
    op: str
    args: Tuple[object, ...] = field(default_factory=tuple)

    @classmethod
    def ref(cls, name: str) -> "Expr":
        return cls("REF", (normalize_name(name),))

    def dependencies(self) -> Set[str]:
        out: Set[str] = set()
        if self.op == "REF":
            out.add(str(self.args[0]))
            return out
        for arg in self.args:
            if isinstance(arg, Expr):
                out |= arg.dependencies()
        return out

    def pretty(self) -> str:
        if self.op == "REF":
            return str(self.args[0])
        if self.op == "AND":
            return " AND ".join(a.pretty() if isinstance(a, Expr) else str(a) for a in self.args)
        if self.op in {"MUL", "DIV", "ADD", "SUB"} and len(self.args) == 2:
            symbol = {"MUL":"×", "DIV":"÷", "ADD":"+", "SUB":"−"}[self.op]
            return f"({self.args[0].pretty()} {symbol} {self.args[1].pretty()})"
        if self.op == "RATE" and len(self.args) == 2:
            return f"rate_of_change({self.args[0].pretty()}, {self.args[1].pretty()})"
        if self.op == "IS_A" and len(self.args) == 1:
            return f"is_a({self.args[0].pretty()})"
        return f"{self.op}({', '.join(a.pretty() if isinstance(a, Expr) else str(a) for a in self.args)})"

    def to_dict(self) -> Dict[str, object]:
        return {"op": self.op, "args": [a.to_dict() if isinstance(a, Expr) else a for a in self.args]}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Expr":
        args = []
        for a in data.get("args", []):
            if isinstance(a, dict) and "op" in a:
                args.append(cls.from_dict(a))
            else:
                args.append(a)
        return cls(str(data["op"]), tuple(args))


@dataclass
class ConceptRecord:
    name: str
    kind: str
    definition: Optional[Expr] = None
    source_sentence: str = ""
    primitive: bool = False
    grounded: bool = False
    support: int = 0

    def dependencies(self) -> Set[str]:
        return set() if self.definition is None else self.definition.dependencies()

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["definition"] = None if self.definition is None else self.definition.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ConceptRecord":
        definition = data.get("definition")
        return cls(
            name=str(data["name"]),
            kind=str(data.get("kind", "defined")),
            definition=None if definition is None else Expr.from_dict(definition),
            source_sentence=str(data.get("source_sentence", "")),
            primitive=bool(data.get("primitive", False)),
            grounded=bool(data.get("grounded", False)),
            support=int(data.get("support", 0)),
        )


def normalize_name(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    text = re.sub(r"^(?:a|an|the)\s+", "", text)
    return text.strip(" .,:;!?\t\n")


class DefinitionParseError(ValueError):
    pass


class DefinitionParser:
    """Generic construction parser for concept-from-concept definitions."""

    PRODUCT = re.compile(r"^(?:a\s+|an\s+)?(.+?)\s+is\s+(?:the\s+)?product\s+of\s+(.+?)\s+and\s+(.+?)\.?$", re.I)
    MULTIPLIED = re.compile(r"^(?:a\s+|an\s+)?(.+?)\s+is\s+(.+?)\s+(?:multiplied\s+by|times)\s+(.+?)\.?$", re.I)
    DIVIDED = re.compile(r"^(?:a\s+|an\s+)?(.+?)\s+is\s+(.+?)\s+(?:divided\s+by|per)\s+(.+?)\.?$", re.I)
    RATE = re.compile(r"^(?:a\s+|an\s+)?(.+?)\s+is\s+(?:the\s+)?rate\s+of\s+change\s+of\s+(.+?)\s+(?:with\s+respect\s+to|over)\s+(.+?)\.?$", re.I)
    MEANS = re.compile(r"^(.+?)\s+(?:means|is\s+defined\s+as)\s+(.+?)\.?$", re.I)
    YOUNG_KIND = re.compile(r"^(?:a\s+|an\s+)?(.+?)\s+is\s+(?:a\s+|an\s+)?(.+?)\s+(.+?)\.?$", re.I)
    ISA = re.compile(r"^(?:a\s+|an\s+)?(.+?)\s+is\s+(?:a\s+|an\s+)?(.+?)\.?$", re.I)

    def parse(self, sentence: str) -> Tuple[str, Expr, str]:
        s = re.sub(r"\s+", " ", sentence.strip())
        m = self.PRODUCT.match(s) or self.MULTIPLIED.match(s)
        if m:
            return normalize_name(m.group(1)), Expr("MUL", (Expr.ref(m.group(2)), Expr.ref(m.group(3)))), "derived_quantity"
        m = self.DIVIDED.match(s)
        if m:
            return normalize_name(m.group(1)), Expr("DIV", (Expr.ref(m.group(2)), Expr.ref(m.group(3)))), "derived_quantity"
        m = self.RATE.match(s)
        if m:
            return normalize_name(m.group(1)), Expr("RATE", (Expr.ref(m.group(2)), Expr.ref(m.group(3)))), "derived_quantity"
        m = self.MEANS.match(s)
        if m:
            return normalize_name(m.group(1)), Expr.ref(m.group(2)), "alias"
        m = self.YOUNG_KIND.match(s)
        if m:
            lhs = normalize_name(m.group(1))
            first, second = normalize_name(m.group(2)), normalize_name(m.group(3))
            if lhs and first and second and first != "defined":
                return lhs, Expr("AND", (Expr.ref(first), Expr.ref(second))), "composite"
        m = self.ISA.match(s)
        if m:
            return normalize_name(m.group(1)), Expr("IS_A", (Expr.ref(m.group(2)),)), "category"
        raise DefinitionParseError(f"definition construction not understood: {sentence!r}")


class ConceptStore:
    VERSION = "APCN-V0.10-CONCEPT-STORE"

    def __init__(self):
        self.records: Dict[str, ConceptRecord] = {}
        self.parser = DefinitionParser()
        self.definition_count = 0

    def add_primitive(self, name: str, kind: str = "primitive", grounded: bool = True) -> ConceptRecord:
        name = normalize_name(name)
        rec = self.records.get(name)
        if rec is None:
            rec = ConceptRecord(name=name, kind=kind, primitive=True, grounded=grounded, support=1)
            self.records[name] = rec
        else:
            rec.primitive = True
            rec.grounded = rec.grounded or grounded
            rec.support += 1
        return rec

    def learn_definition(self, sentence: str) -> ConceptRecord:
        name, expr, kind = self.parser.parse(sentence)
        if name in expr.dependencies():
            raise DefinitionParseError("a concept cannot directly define itself")
        old = self.records.get(name)
        support = 1 if old is None else old.support + 1
        rec = ConceptRecord(name=name, kind=kind, definition=expr, source_sentence=sentence.strip(), primitive=False, grounded=False, support=support)
        self.records[name] = rec
        self.definition_count += 1
        if self._has_cycle(name):
            if old is None:
                del self.records[name]
            else:
                self.records[name] = old
            raise DefinitionParseError(f"definition would create a dependency cycle for {name!r}")
        return rec

    def _has_cycle(self, start: str) -> bool:
        visiting: Set[str] = set()
        visited: Set[str] = set()
        def dfs(name: str) -> bool:
            if name in visiting:
                return True
            if name in visited:
                return False
            visited.add(name)
            rec = self.records.get(name)
            if rec is None or rec.definition is None:
                return False
            visiting.add(name)
            for dep in rec.dependencies():
                if dep in self.records and dfs(dep):
                    return True
            visiting.remove(name)
            return False
        return dfs(start)

    def unresolved_dependencies(self, name: str, recursive: bool = True) -> Set[str]:
        name = normalize_name(name)
        seen: Set[str] = set()
        unresolved: Set[str] = set()
        def walk(n: str) -> None:
            if n in seen:
                return
            seen.add(n)
            rec = self.records.get(n)
            if rec is None:
                unresolved.add(n); return
            if rec.primitive:
                if not rec.grounded:
                    unresolved.add(n)
                return
            if rec.definition is None:
                unresolved.add(n); return
            for dep in rec.dependencies():
                if dep not in self.records:
                    unresolved.add(dep)
                elif recursive:
                    walk(dep)
        walk(name)
        unresolved.discard(name)
        return unresolved

    def understanding(self, name: str) -> Dict[str, object]:
        name = normalize_name(name)
        rec = self.records.get(name)
        if rec is None:
            return {"name": name, "known": False, "complete": False, "unresolved": [name], "depth": 0}
        unresolved = sorted(self.unresolved_dependencies(name))
        return {
            "name": name,
            "known": True,
            "kind": rec.kind,
            "definition": None if rec.definition is None else rec.definition.pretty(),
            "dependencies": sorted(rec.dependencies()),
            "unresolved": unresolved,
            "complete": len(unresolved) == 0 and (rec.primitive and rec.grounded or rec.definition is not None),
            "depth": self.dependency_depth(name),
            "support": rec.support,
        }

    def dependency_depth(self, name: str) -> int:
        name = normalize_name(name)
        memo: Dict[str, int] = {}
        def depth(n: str) -> int:
            if n in memo:
                return memo[n]
            rec = self.records.get(n)
            if rec is None or rec.primitive or rec.definition is None:
                memo[n] = 0; return 0
            d = 1 + max((depth(dep) for dep in rec.dependencies()), default=0)
            memo[n] = d
            return d
        return depth(name)

    def evaluate(self, name: str, values: Mapping[str, float]) -> float:
        values_n = {normalize_name(k): float(v) for k, v in values.items()}
        cache: Dict[str, float] = {}
        def eval_name(n: str) -> float:
            n = normalize_name(n)
            if n in cache:
                return cache[n]
            if n in values_n:
                cache[n] = values_n[n]; return cache[n]
            rec = self.records.get(n)
            if rec is None:
                raise KeyError(f"missing concept/value {n!r}")
            if rec.primitive:
                raise KeyError(f"primitive concept {n!r} requires a supplied numeric value")
            if rec.definition is None:
                raise KeyError(f"concept {n!r} has no executable definition")
            val = eval_expr(rec.definition); cache[n] = val; return val
        def eval_expr(expr: Expr) -> float:
            if expr.op == "REF": return eval_name(str(expr.args[0]))
            if expr.op == "MUL": return eval_expr(expr.args[0]) * eval_expr(expr.args[1])
            if expr.op == "DIV":
                denominator = eval_expr(expr.args[1])
                if abs(denominator) < 1e-15: raise ZeroDivisionError("definition evaluation divided by zero")
                return eval_expr(expr.args[0]) / denominator
            if expr.op == "ADD": return eval_expr(expr.args[0]) + eval_expr(expr.args[1])
            if expr.op == "SUB": return eval_expr(expr.args[0]) - eval_expr(expr.args[1])
            if expr.op == "RATE":
                denominator = eval_expr(expr.args[1])
                if abs(denominator) < 1e-15: raise ZeroDivisionError("rate definition divided by zero")
                return eval_expr(expr.args[0]) / denominator
            raise TypeError(f"expression {expr.op!r} is structural, not numeric")
        return float(eval_name(name))

    def save(self, path: str | Path) -> None:
        data = {"version": self.VERSION, "definition_count": self.definition_count, "records": {k: v.to_dict() for k, v in self.records.items()}}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ConceptStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls()
        obj.definition_count = int(data.get("definition_count", 0))
        obj.records = {k: ConceptRecord.from_dict(v) for k, v in data.get("records", {}).items()}
        return obj


class DefinitionCurriculum:
    PRIMITIVES = (
        "distance", "time", "mass", "volume", "velocity", "velocity change",
        "work", "area", "amount", "duration", "young", "dog", "animal",
    )
    DEFINITIONS = (
        "speed is distance divided by time",
        "density is mass divided by volume",
        "momentum is the product of mass and velocity",
        "acceleration is velocity change divided by time",
        "force is the product of mass and acceleration",
        "power is work divided by time",
        "pressure is force divided by area",
        "puppy is young dog",
    )

    def __init__(self, store: Optional[ConceptStore] = None):
        self.store = store or ConceptStore()
        self.index = 0
        for p in self.PRIMITIVES:
            self.store.add_primitive(p, grounded=True)

    def step(self) -> ConceptRecord:
        sentence = self.DEFINITIONS[self.index % len(self.DEFINITIONS)]
        self.index += 1
        return self.store.learn_definition(sentence)

    def train_all_once(self) -> List[ConceptRecord]:
        return [self.step() for _ in range(len(self.DEFINITIONS))]
