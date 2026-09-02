from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random

from .semantic import EntityRef, SemanticNode


@dataclass(frozen=True)
class Lexicon:
    colors: Dict[str, str]
    shapes: Dict[str, str]
    relations: Dict[str, str]
    operators: Dict[str, str]
    verbs: Dict[str, Tuple[str, ...]]

    @classmethod
    def english(cls) -> "Lexicon":
        return cls(
            colors={"C0": "yellow", "C1": "red", "C2": "green", "C3": "blue", "C4": "purple"},
            shapes={"S0": "circle", "S1": "square", "S2": "triangle", "S3": "rectangle"},
            relations={"R0": "inside", "R1": "left of", "R2": "above"},
            operators={"GROUP": "and", "SEQUENCE": "then", "NEGATE": "not", "FOCUS_REF": "it"},
            verbs={"GOAL": ("put", "place")},
        )

    @classmethod
    def scrambled(cls) -> "Lexicon":
        """Arbitrary vocabulary used to verify that literal English keys are not required."""
        return cls(
            colors={"C0": "dax", "C1": "mip", "C2": "sorn", "C3": "tave", "C4": "plin"},
            shapes={"S0": "blicket", "S1": "koba", "S2": "naru", "S3": "vemi"},
            relations={"R0": "zorp", "R1": "fen lo", "R2": "kesh"},
            operators={"GROUP": "ka", "SEQUENCE": "vo", "NEGATE": "nu", "FOCUS_REF": "ti"},
            verbs={"GOAL": ("mek", "dor")},
        )


@dataclass
class LanguageEpisode:
    utterance: str
    program: SemanticNode
    discourse_focus: Optional[EntityRef] = None
    phase: str = "lexical"
    held_out_template: bool = False


class SemanticTeacher:
    """Procedural language/world teacher for V0.9.

    It generates a language utterance paired with a language-independent semantic
    program. The program stands in for the world-model interpretation obtained
    from perception/demonstration. The learner never receives the Lexicon maps.
    """

    ASSERT_TRAIN = ("the {a} is {rel} the {b}", "{a} is {rel} {b}")
    ASSERT_TEST = ("the {a} sits {rel} the {b}", "observe {a} {rel} {b}")
    QUERY_TRAIN = ("is the {a} {rel} the {b}", "is {a} {rel} {b}")
    QUERY_TEST = ("is it true that {a} is {rel} {b}",)
    GOAL_TRAIN = ("{verb} the {a} {rel} the {b}", "{verb} {a} {rel} {b}")
    GOAL_TEST = ("please {verb} the {a} {rel} the {b}",)

    def __init__(self, seed: int = 9, lexicon: Optional[Lexicon] = None):
        self.rng = random.Random(seed)
        self.lexicon = lexicon or Lexicon.english()
        self.colors = tuple(self.lexicon.colors)
        self.shapes = tuple(self.lexicon.shapes)
        self.relations = tuple(self.lexicon.relations)

    def entity(self, color: Optional[str] = None, shape: Optional[str] = None, instance: int = 0) -> EntityRef:
        return EntityRef(color or self.rng.choice(self.colors), shape or self.rng.choice(self.shapes), instance)

    def _entity_text(self, e: EntityRef) -> str:
        return f"{self.lexicon.colors[e.color]} {self.lexicon.shapes[e.shape]}"

    def _rel_text(self, rel: str) -> str:
        return self.lexicon.relations[rel]

    def _verb(self) -> str:
        return self.rng.choice(self.lexicon.verbs["GOAL"])

    def simple(self, intent: Optional[str] = None, relation: Optional[str] = None, subject: Optional[EntityRef] = None, object: Optional[EntityRef] = None, held_out: bool = False, phase: str = "simple") -> LanguageEpisode:
        intent = intent or self.rng.choice(("ASSERT", "QUERY", "GOAL"))
        relation = relation or self.rng.choice(self.relations)
        subject = subject or self.entity()
        object = object or self.entity(instance=1)
        while object.color == subject.color and object.shape == subject.shape:
            object = self.entity(instance=1)
        a, b = self._entity_text(subject), self._entity_text(object)
        if intent == "ASSERT":
            template = self.rng.choice(self.ASSERT_TEST if held_out else self.ASSERT_TRAIN)
            utterance = template.format(a=a, b=b, rel=self._rel_text(relation))
        elif intent == "QUERY":
            template = self.rng.choice(self.QUERY_TEST if held_out else self.QUERY_TRAIN)
            utterance = template.format(a=a, b=b, rel=self._rel_text(relation))
        else:
            template = self.rng.choice(self.GOAL_TEST if held_out else self.GOAL_TRAIN)
            utterance = template.format(a=a, b=b, rel=self._rel_text(relation), verb=self._verb())
        return LanguageEpisode(utterance, SemanticNode.relation_node(relation, subject, object, intent=intent), phase=phase, held_out_template=held_out)

    def group(self, held_out: bool = False) -> LanguageEpisode:
        rel = self.rng.choice(self.relations)
        a, c, b = self.entity(), self.entity(), self.entity(instance=1)
        while c.key() == a.key():
            c = self.entity()
        utterance = f"{self._verb()} the {self._entity_text(a)} {self.lexicon.operators['GROUP']} the {self._entity_text(c)} {self._rel_text(rel)} the {self._entity_text(b)}"
        p1 = SemanticNode.relation_node(rel, a, b, intent="GOAL")
        p2 = SemanticNode.relation_node(rel, c, b, intent="GOAL")
        return LanguageEpisode(utterance, SemanticNode("GROUP", children=(p1, p2)), phase="operators", held_out_template=held_out)

    def sequence(self, held_out: bool = False) -> LanguageEpisode:
        first = self.simple(intent="GOAL", held_out=False, phase="operators")
        second = self.simple(intent="GOAL", held_out=False, phase="operators")
        utterance = f"{first.utterance} {self.lexicon.operators['SEQUENCE']} {second.utterance}"
        return LanguageEpisode(utterance, SemanticNode("SEQUENCE", children=(first.program, second.program)), phase="operators", held_out_template=held_out)

    def negated(self) -> LanguageEpisode:
        base = self.simple(intent="ASSERT", held_out=False, phase="operators")
        rel = base.program.relations()[0]
        phrase = self._rel_text(rel)
        utterance = base.utterance.replace(phrase, f"{self.lexicon.operators['NEGATE']} {phrase}", 1)
        return LanguageEpisode(utterance, SemanticNode("NEGATE", children=(base.program,)), phase="operators")

    def reference_pair(self) -> Tuple[LanguageEpisode, LanguageEpisode]:
        first = self.simple(intent="GOAL", held_out=False, phase="reference")
        focus = first.program.atom().subject
        obj = self.entity(instance=2)
        rel = self.rng.choice(self.relations)
        utterance = f"{self._verb()} {self.lexicon.operators['FOCUS_REF']} {self._rel_text(rel)} the {self._entity_text(obj)}"
        second = LanguageEpisode(utterance, SemanticNode.relation_node(rel, focus, obj, intent="GOAL"), discourse_focus=focus, phase="reference")
        return first, second

    def curriculum_episode(self, index: int) -> List[LanguageEpisode]:
        if index < 400:
            return [self.simple(intent="ASSERT", phase="lexical")]
        if index < 800:
            return [self.simple(intent=self.rng.choice(("ASSERT", "QUERY", "GOAL")), phase="intent")]
        if index < 1200:
            return [self.group() if self.rng.random() < 0.45 else self.simple(intent="GOAL", phase="composition")]
        if index < 1600:
            r = self.rng.random()
            return [self.sequence() if r < 0.40 else self.negated() if r < 0.70 else self.simple(phase="operators")]
        if self.rng.random() < 0.35:
            return list(self.reference_pair())
        return [self.simple(held_out=False, phase="mixed")]
