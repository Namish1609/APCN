from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import math

from apcn_v10.language_common import tokenize
from .dialogue_learning import ConversationTeacherV15, DialogueActLearner, DialogueEpisode


class ConversationTeacherV151(ConversationTeacherV15):
    """Calibrated V0.15 teacher with lexical bridge experiences.

    Bridge sentences expose semantic vocabulary in constructions that remain
    different from the held-out test sentences. This tests recombination rather
    than expecting zero-shot inference of both new vocabulary and new syntax.
    """

    TRAIN = dict(ConversationTeacherV15.TRAIN)
    TRAIN.update({
        "DEFINE": ConversationTeacherV15.TRAIN["DEFINE"] + (
            "define the idea {a} for me",
            "could you define {a} for me",
            "what definition do you use for {a}",
            "describe what {a} means in your knowledge",
        ),
        "DEPS": ConversationTeacherV15.TRAIN["DEPS"] + (
            "which ideas support {a}",
            "what is {a} conceptually based on",
            "what concepts sit underneath {a}",
        ),
        "KNOW": ConversationTeacherV15.TRAIN["KNOW"] + (
            "do you understand the idea {a}",
            "can you say you know {a}",
        ),
        "ABOUT": ConversationTeacherV15.TRAIN["ABOUT"] + (
            "what can you tell me about {a}",
            "give me a broader explanation of {a}",
        ),
        "COMPARE": ConversationTeacherV15.TRAIN["COMPARE"] + (
            "what makes {a} different from {b}",
            "set {a} against {b} and explain the contrast",
            "set {a} next to {b} for comparison",
        ),
        "FOLLOW_DEPS": ConversationTeacherV15.TRAIN["FOLLOW_DEPS"] + (
            "what is it based on",
            "which ideas are underneath it",
            "what concepts feed that result",
        ),
        "FOLLOW_WHY": ConversationTeacherV15.TRAIN["FOLLOW_WHY"] + (
            "where does that answer come from",
            "what supports that conclusion",
        ),
        "FOLLOW_MORE": ConversationTeacherV15.TRAIN["FOLLOW_MORE"] + (
            "continue the explanation of that",
            "take that explanation further",
        ),
        "LAST_TAUGHT": ConversationTeacherV15.TRAIN["LAST_TAUGHT"] + (
            "which teaching was most recent",
            "remind me of my latest lesson to you",
        ),
        "TOPIC": ConversationTeacherV15.TRAIN["TOPIC"] + (
            "which subject is current",
            "what is the present topic",
        ),
        "HELP": ConversationTeacherV15.TRAIN["HELP"] + (
            "what conversation can we have",
            "which kinds of things can I teach",
        ),
        "GREETING": ConversationTeacherV15.TRAIN["GREETING"] + (
            "greetings friend",
            "nice to talk with you",
        ),
    })


class DialogueActLearnerV151(DialogueActLearner):
    """Dialogue learner with content filtering and learned anchor cues.

    A high-purity learned content word/short phrase can act as an anchor before
    weaker overlapping phrase evidence is aggregated. The anchor is discovered
    from cue statistics; there is no hand-written word -> dialogue-act map.
    """

    STOP = {
        "a","an","the","is","are","was","were","be","been","being",
        "i","me","my","you","your","we","our","it","its","that","this",
        "what","which","who","how","do","does","did","can","could","would","should",
        "and","or","of","to","for","on","in","with","from","at","by","as",
        "please","then","now","just","some","any","there","here",
    }

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
            for i in range(len(toks)-n+1):
                phrase = " ".join(toks[i:i+n])
                if not cls._informative(phrase):
                    continue
                rows.add("A:" + phrase)
                if i == 0:
                    rows.add("S:" + phrase)
                if i+n == len(toks):
                    rows.add("E:" + phrase)
        return sorted(rows)

    def _best_anchor(self, text: str) -> Optional[Tuple[str, float, str]]:
        """Return a learned high-purity content anchor if one exists."""
        candidates = []
        total_all = max(1, sum(self.act_totals.values()))
        for cue in self._cues(text):
            row = self.cue_act.get(cue)
            if not row:
                continue
            support = sum(row.values())
            if support < 3:
                continue
            act, count = row.most_common(1)[0]
            purity = count / support
            baseline = self.act_totals.get(act, 0) / total_all
            discrimination = purity - baseline
            phrase = cue[2:]
            content_tokens = [
                t for t in phrase.split()
                if t not in self.STOP and not t.startswith("conceptslot")
            ]
            if not (1 <= len(content_tokens) <= 2):
                continue
            if purity < .86 or discrimination < .45:
                continue
            reliability = purity * (1.0 - math.exp(-support / 3.0))
            rank = reliability * (1.05 if len(content_tokens) == 1 else 1.0)
            candidates.append((rank, support, act, reliability, cue))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        _, _, act, reliability, cue = candidates[0]
        return act, float(reliability), cue

    def predict(self, utterance: str, concepts: Sequence[str] = ()):
        text = self._replace_concepts(utterance, concepts)
        anchor = self._best_anchor(text)
        if anchor is not None:
            act, reliability, cue = anchor
            confidence = min(.94, .58 + .40 * reliability)
            return act, float(confidence), [(cue, float(reliability))]
        return super().predict(utterance, concepts)


def balanced_bootstrap_dialogue(
    learner: DialogueActLearnerV151,
    teacher: ConversationTeacherV151,
    *,
    repeats_per_template: int = 4,
) -> Dict[str, object]:
    """Guarantee minimum support for every TRAIN construction, never TEST forms.

    Each dialogue act receives the same number of bootstrap observations. Pools
    with fewer templates cycle through them, so act priors stay balanced while
    every training construction receives at least ``repeats_per_template``
    observations. This removes random curriculum coverage as a startup failure
    mode without leaking any held-out wording.
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
