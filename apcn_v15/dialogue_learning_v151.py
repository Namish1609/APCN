from __future__ import annotations

from typing import Dict, List

from apcn_v10.language_common import tokenize
from .dialogue_learning import ConversationTeacherV15, DialogueActLearner, DialogueEpisode


class DialogueActLearnerV151(DialogueActLearner):
    """Small auxiliary dialogue router for the semantic compiler.

    V0.15 course-correction rule: this learner may help identify a dialogue
    operation, but it is not the knowledge store and it must not grow into a
    general text model. It therefore keeps only bounded cue->act statistics,
    excludes low-information function-word-only cues, and stores no raw text.
    """

    VERSION = "APCN-V0.15.1-AUX-DIALOGUE-ROUTER"

    STOP = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "i", "me", "my", "you", "your", "we", "our", "it", "its", "that", "this",
        "what", "which", "who", "how", "do", "does", "did", "can", "could", "would", "should",
        "and", "or", "of", "to", "for", "on", "in", "with", "from", "at", "by", "as",
        "please", "then", "now", "just", "some", "any", "there", "here",
    }

    def __init__(self, max_cues: int = 12000):
        super().__init__(max_cues=max_cues)

    @classmethod
    def _informative(cls, phrase: str) -> bool:
        toks = phrase.split()
        content = [t for t in toks if t not in cls.STOP and not t.startswith("conceptslot")]
        return bool(content)

    @classmethod
    def _cues(cls, text: str, max_n: int = 4) -> List[str]:
        toks = tokenize(text)
        rows = set()
        for n in range(1, min(max_n, len(toks)) + 1):
            for i in range(len(toks) - n + 1):
                phrase = " ".join(toks[i:i+n])
                if not cls._informative(phrase):
                    continue
                rows.add("A:" + phrase)
                if i == 0:
                    rows.add("S:" + phrase)
                if i + n == len(toks):
                    rows.add("E:" + phrase)
        return sorted(rows)


def balanced_bootstrap_dialogue(
    learner: DialogueActLearnerV151,
    teacher: ConversationTeacherV15,
    *,
    repeats_per_template: int = 4,
) -> Dict[str, object]:
    """Give every TRAIN construction minimum support without using TEST forms.

    This removes random startup coverage as a failure mode. It does not add
    lexical bridges after inspecting a particular test failure, and it never
    samples ``teacher.TEST``. After bootstrap, normal training may continue with
    random TRAIN episodes.
    """
    repeats_per_template = max(1, int(repeats_per_template))
    max_templates = max(len(teacher.TRAIN[act]) for act in teacher.acts)
    per_act = max_templates * repeats_per_template
    concepts = teacher.CONCEPTS
    before = learner.observations

    for act_index, act in enumerate(teacher.acts):
        pool = teacher.TRAIN[act]
        for i in range(per_act):
            template = pool[i % len(pool)]
            a = concepts[(i + act_index) % len(concepts)]
            b = concepts[(i + act_index + 1) % len(concepts)]
            if b == a:
                b = concepts[(i + act_index + 2) % len(concepts)]
            utterance = template.format(a=a, b=b)
            if act == "COMPARE":
                slots = (a, b)
            elif act in {"DEFINE", "DEPS", "KNOW", "ABOUT"}:
                slots = (a,)
            else:
                slots = ()
            learner.observe(DialogueEpisode(utterance, act, slots, False))

    return {
        "observations_added": learner.observations - before,
        "per_act": per_act,
        "acts": len(teacher.acts),
        "held_out_examples_used": 0,
        "summary": learner.summary(12),
    }
