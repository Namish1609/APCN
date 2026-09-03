from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class EntityRef:
    color: str
    shape: str
    instance: int = 0

    def key(self) -> str:
        return f"{self.color}:{self.shape}:{self.instance}"

    def to_dict(self) -> Dict[str, object]:
        return {"color": self.color, "shape": self.shape, "instance": self.instance}


@dataclass(frozen=True)
class SemanticNode:
    op: str
    relation: Optional[str] = None
    subject: Optional[EntityRef] = None
    object: Optional[EntityRef] = None
    children: Tuple["SemanticNode", ...] = field(default_factory=tuple)

    @classmethod
    def relation_node(
        cls,
        relation: str,
        subject: EntityRef,
        object: EntityRef,
        intent: str = "ASSERT",
    ) -> "SemanticNode":
        atom = cls("RELATION", relation=relation, subject=subject, object=object)
        return cls(intent, children=(atom,))

    def atom(self) -> Optional["SemanticNode"]:
        if self.op == "RELATION":
            return self
        if len(self.children) == 1 and self.children[0].op == "RELATION":
            return self.children[0]
        return None

    def walk(self) -> Iterable["SemanticNode"]:
        yield self
        for child in self.children:
            yield from child.walk()

    def features(self) -> List[str]:
        out: List[str] = []
        for node in self.walk():
            if node.op in {"ASSERT", "QUERY", "GOAL"}:
                out.append(f"intent:{node.op}")
            elif node.op in {"GROUP", "SEQUENCE", "NEGATE"}:
                out.append(f"operator:{node.op}")
            if node.op == "RELATION" and node.relation:
                out.append(f"relation:{node.relation}")
                if node.subject is not None:
                    out.extend((f"color:{node.subject.color}", f"shape:{node.subject.shape}"))
                if node.object is not None:
                    out.extend((f"color:{node.object.color}", f"shape:{node.object.shape}"))
        return out

    def intent(self) -> Optional[str]:
        for node in self.walk():
            if node.op in {"ASSERT", "QUERY", "GOAL"}:
                return node.op
        return None

    def relations(self) -> List[str]:
        return [n.relation for n in self.walk() if n.op == "RELATION" and n.relation is not None]

    def operators(self) -> List[str]:
        return [n.op for n in self.walk() if n.op in {"GROUP", "SEQUENCE", "NEGATE"}]

    def canonical(self) -> Tuple[object, ...]:
        return (
            self.op,
            self.relation,
            None if self.subject is None else (self.subject.color, self.subject.shape),
            None if self.object is None else (self.object.color, self.object.shape),
            tuple(c.canonical() for c in self.children),
        )

    def pretty(self, indent: int = 0) -> str:
        pad = "  " * indent
        if self.op == "RELATION":
            s = self.subject.key() if self.subject else "?"
            o = self.object.key() if self.object else "?"
            return f"{pad}{self.relation}({s}, {o})"
        lines = [f"{pad}{self.op}"]
        lines.extend(c.pretty(indent + 1) for c in self.children)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        return {
            "op": self.op,
            "relation": self.relation,
            "subject": None if self.subject is None else self.subject.to_dict(),
            "object": None if self.object is None else self.object.to_dict(),
            "children": [c.to_dict() for c in self.children],
        }


def semantic_equal(a: Optional[SemanticNode], b: Optional[SemanticNode]) -> bool:
    return a is not None and b is not None and a.canonical() == b.canonical()
