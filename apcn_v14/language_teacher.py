from __future__ import annotations

from typing import List

from apcn_v10.language_common import LanguageEpisode
from apcn_v10.semantic import SemanticNode
from apcn_v11.language_teacher import RichSemanticTeacherV11


class RichSemanticTeacherV14(RichSemanticTeacherV11):
    """Language-first V0.14 procedural teacher.

    V0.14 deliberately increases surface-form diversity while keeping the same
    language-independent semantic programs. The learner sees only utterance +
    semantic demonstration during learning; these templates are teacher-side
    ground truth and are not visible to the parser.
    """

    ASSERT_V14_TRAIN = (
        "from what you see, the {a} is {rel} the {b}",
        "the relation here is that the {a} is {rel} the {b}",
        "I observe the {a} {rel} the {b}",
        "visually, the {a} appears {rel} the {b}",
        "record that the {a} is {rel} the {b}",
    )
    ASSERT_V14_TEST = (
        "according to the scene, the {a} is {rel} the {b}",
        "the scene indicates that {a} is {rel} {b}",
        "my observation is that the {a} lies {rel} the {b}",
    )
    QUERY_V14_TRAIN = (
        "work out if the {a} is {rel} the {b}?",
        "tell me if the {a} is {rel} the {b}?",
        "do we have the {a} {rel} the {b}?",
        "determine whether the {a} is {rel} the {b}?",
        "check the claim that the {a} is {rel} the {b}?",
    )
    QUERY_V14_TEST = (
        "could you verify that the {a} is {rel} the {b}?",
        "would the scene support the claim that {a} is {rel} {b}?",
        "please establish whether the {a} lies {rel} the {b}?",
    )
    GOAL_V14_TRAIN = (
        "arrange the {a} {rel} the {b}",
        "I want the {a} {rel} the {b}",
        "make sure the {a} ends up {rel} the {b}",
        "cause the {a} to be {rel} the {b}",
        "set things so the {a} is {rel} the {b}",
    )
    GOAL_V14_TEST = (
        "have the {a} end up {rel} the {b}",
        "make the final state place {a} {rel} {b}",
        "the desired result is for the {a} to be {rel} the {b}",
    )
    GROUP_V14_TRAIN = (
        "arrange both the {a} and the {c} {rel} the {b}",
        "I want the {a} plus the {c} {rel} the {b}",
        "make the {a} as well as the {c} be {rel} the {b}",
    )
    GROUP_V14_TEST = (
        "have each of {a} and {c} end up {rel} {b}",
        "the desired state puts both {a} and {c} {rel} {b}",
    )
    SEQUENCE_V14_TRAIN = ("afterwards", "next do", "followed by")
    SEQUENCE_V14_TEST = ("subsequently", "once that is done", "and afterwards")

    def v14_simple(self, intent: str | None = None, *, held_out: bool = False,
                   skill: str = "intent") -> LanguageEpisode:
        intent = intent or self.rng.choice(("ASSERT", "QUERY", "GOAL"))
        rel = self.rng.choice(self.relations)
        a, b = self._distinct_pair()
        data = {"a": self._entity_text(a), "b": self._entity_text(b),
                "rel": self._relation_text(rel)}
        if intent == "ASSERT":
            pool = self.ASSERT_V14_TEST if held_out else self.ASSERT_V14_TRAIN
        elif intent == "QUERY":
            pool = self.QUERY_V14_TEST if held_out else self.QUERY_V14_TRAIN
        elif intent == "GOAL":
            pool = self.GOAL_V14_TEST if held_out else self.GOAL_V14_TRAIN
        else:
            raise ValueError(intent)
        return LanguageEpisode(self.rng.choice(pool).format(**data),
                               SemanticNode.relation_node(rel, a, b, intent),
                               skill, held_out_template=held_out)

    def v14_group(self, *, held_out: bool = False) -> LanguageEpisode:
        a, b = self._distinct_pair(); c = self.entity(2)
        while (c.color, c.shape) in {(a.color, a.shape), (b.color, b.shape)}:
            c = self.entity(2)
        rel = self.rng.choice(self.relations)
        pool = self.GROUP_V14_TEST if held_out else self.GROUP_V14_TRAIN
        text = self.rng.choice(pool).format(
            a=self._entity_text(a), c=self._entity_text(c), b=self._entity_text(b),
            rel=self._relation_text(rel))
        p1 = SemanticNode.relation_node(rel, a, b, "GOAL")
        p2 = SemanticNode.relation_node(rel, c, b, "GOAL")
        return LanguageEpisode(text, SemanticNode("GROUP", children=(p1, p2)),
                               "group", held_out_template=held_out)

    def v14_sequence(self, *, held_out: bool = False) -> LanguageEpisode:
        first = self.v14_simple("GOAL", held_out=held_out, skill="sequence")
        second = self.v14_simple("GOAL", held_out=held_out, skill="sequence")
        connector = self.rng.choice(self.SEQUENCE_V14_TEST if held_out else self.SEQUENCE_V14_TRAIN)
        return LanguageEpisode(
            f"{first.utterance} {connector} {second.utterance}",
            SemanticNode("SEQUENCE", children=(first.program, second.program)),
            "sequence", held_out_template=held_out)

    def v14_negated(self, *, held_out: bool = False) -> LanguageEpisode:
        base = self.v14_simple("ASSERT", held_out=held_out, skill="negation")
        relation = base.program.relations()[0]
        aliases = sorted(self.RELATIONS[relation], key=len, reverse=True)
        phrase = next((x for x in aliases if x in base.utterance), aliases[0])
        cue = self.rng.choice(("not", "definitely not", "in no case"))
        text = base.utterance.replace(phrase, f"{cue} {phrase}", 1)
        return LanguageEpisode(text, SemanticNode("NEGATE", children=(base.program,)),
                               "negation", held_out_template=held_out)

    def for_skill(self, skill: str, held_out: bool = False) -> List[LanguageEpisode]:
        if skill == "grounding" and self.rng.random() < .55:
            return [self.v14_simple("ASSERT", held_out=held_out, skill=skill)]
        if skill == "intent" and self.rng.random() < .72:
            return [self.v14_simple(held_out=held_out, skill=skill)]
        if skill == "group" and self.rng.random() < .62:
            return [self.v14_group(held_out=held_out)]
        if skill == "sequence" and self.rng.random() < .62:
            return [self.v14_sequence(held_out=held_out)]
        if skill == "negation" and self.rng.random() < .62:
            return [self.v14_negated(held_out=held_out)]
        return super().for_skill(skill, held_out)

    def held_out_construction(self, index: int) -> LanguageEpisode:
        kind = index % 5
        if kind == 0:
            return self.v14_simple("ASSERT", held_out=True, skill="intent")
        if kind == 1:
            return self.v14_simple("QUERY", held_out=True, skill="intent")
        if kind == 2:
            return self.v14_simple("GOAL", held_out=True, skill="intent")
        if kind == 3:
            return self.v14_group(held_out=True)
        return self.v14_negated(held_out=True)
