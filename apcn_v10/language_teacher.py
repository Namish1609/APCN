from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import random

from .semantic import EntityRef, SemanticNode
from .language_common import LanguageEpisode

class RichSemanticTeacher:
    """Procedurally generates a large, varied grounded-language curriculum.

    The learner never receives these maps. They live only on the teacher/world
    side and are used to create language paired with language-independent
    semantic programs. Combinatorics of entities, aliases and templates yield
    millions of possible training sentences without storing a giant corpus.
    """

    COLORS: Dict[str, Tuple[str, ...]] = {
        "C0": ("yellow",), "C1": ("red",), "C2": ("green",),
        "C3": ("blue",), "C4": ("purple",), "C5": ("orange",),
    }
    SHAPES: Dict[str, Tuple[str, ...]] = {
        "S0": ("circle",), "S1": ("square",), "S2": ("triangle",),
        "S3": ("rectangle",), "S4": ("ellipse",),
    }
    RELATIONS: Dict[str, Tuple[str, ...]] = {
        "R0": ("inside", "in", "within"),
        "R1": ("left of", "to the left of"),
        "R2": ("above", "over"),
        "R3": ("right of", "to the right of"),
        "R4": ("below", "under"),
        "R5": ("near", "close to"),
    }

    ASSERT_TRAIN = (
        "the {a} is {rel} the {b}",
        "{a} is {rel} {b}",
        "notice the {a} is {rel} the {b}",
        "in this scene the {a} is {rel} the {b}",
        "the {a} sits {rel} the {b}",
        "observe that the {a} is {rel} the {b}",
        "you can see the {a} {rel} the {b}",
        "the {a} remains {rel} the {b}",
    )
    ASSERT_TEST = (
        "look carefully: the {a} is {rel} the {b}",
        "as shown, {a} is {rel} {b}",
        "it is the case that the {a} is {rel} the {b}",
    )
    QUERY_TRAIN = (
        "is the {a} {rel} the {b}?",
        "is {a} {rel} {b}?",
        "is it true the {a} is {rel} the {b}?",
        "check if the {a} is {rel} the {b}?",
        "tell me whether the {a} is {rel} the {b}?",
        "does the {a} sit {rel} the {b}?",
    )
    QUERY_TEST = (
        "would you say the {a} is {rel} the {b}?",
        "can you determine whether {a} is {rel} {b}?",
        "is it correct that the {a} lies {rel} the {b}?",
    )
    GOAL_TRAIN = (
        "put the {a} {rel} the {b}",
        "place the {a} {rel} the {b}",
        "move the {a} so it is {rel} the {b}",
        "position the {a} {rel} the {b}",
        "please put the {a} {rel} the {b}",
        "make the {a} be {rel} the {b}",
    )
    GOAL_TEST = (
        "please place {a} so that it is {rel} {b}",
        "move the {a} until it is {rel} the {b}",
        "please put {a} {rel} {b}",
    )

    GROUP_TRAIN = (
        "put the {a} and the {c} {rel} the {b}",
        "place both the {a} and the {c} {rel} the {b}",
        "move the {a} and the {c} so they are {rel} the {b}",
        "put {a} and {c} {rel} {b}",
    )
    GROUP_TEST = (
        "position these two, {a} and {c}, {rel} the {b}",
        "make both {a} and {c} be {rel} {b}",
    )
    SEQUENCE_CONNECTORS = ("then", "and then", "after that")
    NEGATION_CUES = ("not", "never")
    REFERENCE_CUES = ("it", "that object")

    def __init__(self, seed: int = 10):
        self.rng = random.Random(seed)
        self.colors = tuple(self.COLORS)
        self.shapes = tuple(self.SHAPES)
        self.relations = tuple(self.RELATIONS)

    def entity(self, instance: int = 0) -> EntityRef:
        return EntityRef(self.rng.choice(self.colors), self.rng.choice(self.shapes), instance)

    def _entity_text(self, e: EntityRef) -> str:
        return f"{self.rng.choice(self.COLORS[e.color])} {self.rng.choice(self.SHAPES[e.shape])}"

    def _relation_text(self, relation: str) -> str:
        return self.rng.choice(self.RELATIONS[relation])

    def _distinct_pair(self) -> Tuple[EntityRef, EntityRef]:
        a, b = self.entity(0), self.entity(1)
        while (a.color, a.shape) == (b.color, b.shape):
            b = self.entity(1)
        return a, b

    def simple(
        self,
        intent: Optional[str] = None,
        held_out: bool = False,
        skill: str = "grounding",
    ) -> LanguageEpisode:
        intent = intent or self.rng.choice(("ASSERT", "QUERY", "GOAL"))
        rel = self.rng.choice(self.relations)
        a, b = self._distinct_pair()
        data = {"a": self._entity_text(a), "b": self._entity_text(b), "rel": self._relation_text(rel)}
        if intent == "ASSERT":
            template = self.rng.choice(self.ASSERT_TEST if held_out else self.ASSERT_TRAIN)
        elif intent == "QUERY":
            template = self.rng.choice(self.QUERY_TEST if held_out else self.QUERY_TRAIN)
        else:
            template = self.rng.choice(self.GOAL_TEST if held_out else self.GOAL_TRAIN)
        return LanguageEpisode(
            template.format(**data),
            SemanticNode.relation_node(rel, a, b, intent),
            skill=skill,
            held_out_template=held_out,
        )

    def group(self, held_out: bool = False) -> LanguageEpisode:
        a, b = self._distinct_pair()
        c = self.entity(2)
        while (c.color, c.shape) in {(a.color, a.shape), (b.color, b.shape)}:
            c = self.entity(2)
        rel = self.rng.choice(self.relations)
        template = self.rng.choice(self.GROUP_TEST if held_out else self.GROUP_TRAIN)
        text = template.format(a=self._entity_text(a), c=self._entity_text(c), b=self._entity_text(b), rel=self._relation_text(rel))
        p1 = SemanticNode.relation_node(rel, a, b, "GOAL")
        p2 = SemanticNode.relation_node(rel, c, b, "GOAL")
        return LanguageEpisode(text, SemanticNode("GROUP", children=(p1, p2)), "group", held_out_template=held_out)

    def sequence(self, held_out: bool = False) -> LanguageEpisode:
        first = self.simple(intent="GOAL", held_out=held_out, skill="sequence")
        second = self.simple(intent="GOAL", held_out=held_out, skill="sequence")
        connector = self.rng.choice(self.SEQUENCE_CONNECTORS)
        return LanguageEpisode(
            f"{first.utterance} {connector} {second.utterance}",
            SemanticNode("SEQUENCE", children=(first.program, second.program)),
            "sequence",
            held_out_template=held_out,
        )

    def negated(self, held_out: bool = False) -> LanguageEpisode:
        base = self.simple(intent="ASSERT", held_out=held_out, skill="negation")
        relation = base.program.relations()[0]
        aliases = sorted(self.RELATIONS[relation], key=len, reverse=True)
        phrase = next((x for x in aliases if x in base.utterance), aliases[0])
        cue = self.rng.choice(self.NEGATION_CUES)
        text = base.utterance.replace(phrase, f"{cue} {phrase}", 1)
        return LanguageEpisode(text, SemanticNode("NEGATE", children=(base.program,)), "negation", held_out_template=held_out)

    def reference_pair(self, held_out: bool = False) -> Tuple[LanguageEpisode, LanguageEpisode]:
        first = self.simple(intent="GOAL", held_out=held_out, skill="reference")
        atom = first.program.atom()
        assert atom is not None and atom.subject is not None
        focus = atom.subject
        target = self.entity(2)
        while (target.color, target.shape) == (focus.color, focus.shape):
            target = self.entity(2)
        rel = self.rng.choice(self.relations)
        ref = self.rng.choice(self.REFERENCE_CUES)
        if held_out:
            text = f"now position {ref} {self._relation_text(rel)} the {self._entity_text(target)}"
        else:
            text = self.rng.choice((
                f"now put {ref} {self._relation_text(rel)} the {self._entity_text(target)}",
                f"next place {ref} {self._relation_text(rel)} the {self._entity_text(target)}",
            ))
        second = LanguageEpisode(
            text,
            SemanticNode.relation_node(rel, focus, target, "GOAL"),
            "reference",
            discourse_focus=focus,
            held_out_template=held_out,
        )
        return first, second

    def for_skill(self, skill: str, held_out: bool = False) -> List[LanguageEpisode]:
        if skill == "grounding":
            return [self.simple(intent="ASSERT", held_out=held_out, skill=skill)]
        if skill == "intent":
            return [self.simple(intent=self.rng.choice(("ASSERT", "QUERY", "GOAL")), held_out=held_out, skill=skill)]
        if skill == "group":
            return [self.group(held_out)]
        if skill == "sequence":
            return [self.sequence(held_out)]
        if skill == "negation":
            return [self.negated(held_out)]
        if skill == "reference":
            return list(self.reference_pair(held_out))
        raise ValueError(f"unknown language skill {skill!r}")
