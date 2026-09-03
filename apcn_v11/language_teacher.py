from __future__ import annotations

from typing import List, Tuple

from apcn_v10.language_teacher import RichSemanticTeacher
from apcn_v10.language_common import LanguageEpisode
from apcn_v10.semantic import EntityRef, SemanticNode


class RichSemanticTeacherV11(RichSemanticTeacher):
    """V0.11 teacher with cleaner discourse episodes and composable intent contrasts.

    The teacher owns ground truth. The learner still receives only utterance plus
    the language-independent semantic demonstration during learning.
    """

    ASSERT_PREFIX = (
        "observe that", "notice that", "the scene shows that", "it is the case that",
        "look carefully and note that", "as shown",
    )
    QUERY_PREFIX = (
        "can you determine whether", "would you say", "check whether",
        "decide whether", "is it true that", "can you tell whether",
    )
    GOAL_PREFIX = (
        "please position", "now position", "please place", "move", "put", "set",
    )

    def _new_target_distinct_from(self, refs: Tuple[EntityRef, ...], instance: int) -> EntityRef:
        used = {(r.color, r.shape) for r in refs}
        target = self.entity(instance)
        while (target.color, target.shape) in used:
            target = self.entity(instance)
        return target

    def reference_pair(self, held_out: bool = False) -> Tuple[LanguageEpisode, LanguageEpisode]:
        first = self.simple(intent="GOAL", held_out=held_out, skill="reference")
        atom = first.program.atom()
        assert atom is not None and atom.subject is not None and atom.object is not None
        focus = atom.subject
        target = self._new_target_distinct_from((atom.subject, atom.object), 2)
        rel = self.rng.choice(self.relations)
        rel_text = self._relation_text(rel)
        target_text = self._entity_text(target)
        if held_out:
            templates = (
                f"now position that object {rel_text} the {target_text}",
                f"after that, move it {rel_text} the {target_text}",
                f"next position the same object {rel_text} the {target_text}",
            )
        else:
            templates = (
                f"now put it {rel_text} the {target_text}",
                f"next place that object {rel_text} the {target_text}",
                f"move the same object {rel_text} the {target_text}",
                f"then position it {rel_text} the {target_text}",
            )
        second = LanguageEpisode(
            self.rng.choice(templates),
            SemanticNode.relation_node(rel, focus, target, "GOAL"),
            "reference",
            discourse_focus=focus,
            held_out_template=held_out,
        )
        return first, second

    def intent_contrast(self, intent: str, *, held_out: bool = False) -> LanguageEpisode:
        """Generate a fresh intent example from independently sampled parts.

        This is used by consolidation so the system can ask for contrastive
        evidence without a user selecting sentence templates manually.
        """
        intent = str(intent).upper()
        rel = self.rng.choice(self.relations)
        a, b = self._distinct_pair()
        a_text = self._entity_text(a)
        b_text = self._entity_text(b)
        rel_text = self._relation_text(rel)
        if intent == "ASSERT":
            prefix = self.rng.choice(self.ASSERT_PREFIX)
            if prefix == "as shown":
                text = f"{prefix}, {a_text} is {rel_text} {b_text}"
            else:
                text = f"{prefix} the {a_text} is {rel_text} the {b_text}"
        elif intent == "QUERY":
            prefix = self.rng.choice(self.QUERY_PREFIX)
            text = f"{prefix} the {a_text} is {rel_text} the {b_text}?"
        elif intent == "GOAL":
            prefix = self.rng.choice(self.GOAL_PREFIX)
            text = f"{prefix} the {a_text} {rel_text} the {b_text}"
        else:
            raise ValueError(f"unsupported intent {intent!r}")
        return LanguageEpisode(text, SemanticNode.relation_node(rel, a, b, intent),
                               "intent", held_out_template=held_out)

    def for_skill(self, skill: str, held_out: bool = False) -> List[LanguageEpisode]:
        if skill == "reference":
            return list(self.reference_pair(held_out))
        return super().for_skill(skill, held_out)
