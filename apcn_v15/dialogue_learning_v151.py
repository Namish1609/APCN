from __future__ import annotations

from typing import List

from apcn_v10.language_common import tokenize
from .dialogue_learning import ConversationTeacherV15, DialogueActLearner


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
    """Dialogue learner that excludes low-information function-word cues."""

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
