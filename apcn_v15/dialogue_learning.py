from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Counter as CounterType, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import random
import re

from apcn_v10.definitions import normalize_name
from apcn_v10.language_common import tokenize


@dataclass(frozen=True)
class DialogueEpisode:
    utterance: str
    act: str
    concepts: Tuple[str, ...] = ()
    held_out: bool = False


class ConversationTeacherV15:
    """Procedural semantic teacher for dialogue constructions.

    The teacher owns surface templates and language-independent dialogue acts.
    The learner never receives a hardcoded phrase -> act lookup table.
    """

    CONCEPTS = ("acceleration", "speed", "density", "force", "pressure", "momentum", "energy")

    TRAIN: Dict[str, Tuple[str, ...]] = {
        "DEFINE": (
            "can you explain {a}",
            "give me the meaning of {a}",
            "I want a definition of {a}",
            "tell me what {a} means",
            "explain the concept {a}",
        ),
        "DEPS": (
            "which concepts does {a} rely on",
            "what inputs define {a}",
            "list the dependencies of {a}",
            "what ideas does {a} depend upon",
        ),
        "KNOW": (
            "are you able to explain {a}",
            "is {a} in your knowledge",
            "do you have an understanding of {a}",
            "are you familiar with the concept {a}",
        ),
        "ABOUT": (
            "say more about {a}",
            "give me information about {a}",
            "tell me some details about {a}",
            "discuss {a} for me",
        ),
        "COMPARE": (
            "contrast {a} with {b}",
            "how do {a} and {b} differ",
            "compare the concepts {a} and {b}",
            "tell me the difference between {a} and {b}",
        ),
        "FOLLOW_DEPS": (
            "and its dependencies",
            "what inputs does that use",
            "what does that rely on",
            "which concepts support it",
        ),
        "FOLLOW_WHY": (
            "what evidence supports that",
            "what led you to that answer",
            "give me the reason for that",
            "what is the basis of that answer",
        ),
        "FOLLOW_MORE": (
            "continue on that",
            "say more about that",
            "go deeper on it",
            "expand that explanation",
        ),
        "LAST_TAUGHT": (
            "remind me what I taught you last",
            "tell me the last thing I taught you",
            "what was my previous teaching",
        ),
        "TOPIC": (
            "remind me of our topic",
            "what topic are we discussing",
            "tell me our current subject",
        ),
        "HELP": (
            "show me what I can ask",
            "tell me how to teach you",
            "what kinds of questions can I ask",
        ),
        "GREETING": (
            "hello there",
            "good to talk with you",
            "hey there",
            "hi apcn",
        ),
    }

    TEST: Dict[str, Tuple[str, ...]] = {
        "DEFINE": (
            "could you give me a definition for {a}",
            "how would you define {a}",
            "describe the meaning of {a}",
        ),
        "DEPS": (
            "which ideas feed into {a}",
            "what is {a} built from conceptually",
            "name the concepts underlying {a}",
        ),
        "KNOW": (
            "would you say you understand {a}",
            "is your understanding of {a} established",
            "can you claim knowledge of {a}",
        ),
        "ABOUT": (
            "what can you tell me concerning {a}",
            "give me a broader account of {a}",
            "talk through what you know about {a}",
        ),
        "COMPARE": (
            "set {a} beside {b} conceptually",
            "in what way are {a} and {b} different",
            "draw a conceptual contrast between {a} and {b}",
        ),
        "FOLLOW_DEPS": (
            "which things feed into it",
            "and what are the concepts underneath that",
            "what is it based on conceptually",
        ),
        "FOLLOW_WHY": (
            "what is your basis for that",
            "what supports the answer you just gave",
            "where does that conclusion come from",
        ),
        "FOLLOW_MORE": (
            "expand on that",
            "continue the explanation",
            "take that idea further",
        ),
        "LAST_TAUGHT": (
            "what was my latest teaching",
            "which thing did I teach most recently",
        ),
        "TOPIC": (
            "what subject are we on",
            "which concept is our present topic",
        ),
        "HELP": (
            "what sort of conversation do you support",
            "show me the ways I can teach you",
        ),
        "GREETING": (
            "greetings apcn",
            "nice to speak with you",
        ),
    }

    def __init__(self, seed: int = 15):
        self.rng = random.Random(seed)
        self.acts = tuple(self.TRAIN)

    def episode(self, act: Optional[str] = None, *, held_out: bool = False) -> DialogueEpisode:
        act = act or self.rng.choice(self.acts)
        pool = self.TEST[act] if held_out else self.TRAIN[act]
        a = self.rng.choice(self.CONCEPTS)
        b = self.rng.choice(tuple(x for x in self.CONCEPTS if x != a))
        text = self.rng.choice(pool).format(a=a, b=b)
        concepts: Tuple[str, ...]
        if "{a}" in self.rng.choice(("", "")):  # unreachable; keeps templates teacher-side only
            concepts = ()
        elif act == "COMPARE":
            concepts = (a, b)
        elif act in {"DEFINE", "DEPS", "KNOW", "ABOUT"}:
            concepts = (a,)
        else:
            concepts = ()
        return DialogueEpisode(text, act, concepts, held_out)


class DialogueActLearner:
    """Sparse non-neural cue learner for conversational semantic operations."""

    VERSION = "APCN-V0.15-DIALOGUE-ACT-LEARNER"

    def __init__(self, max_cues: int = 24000):
        self.max_cues = int(max_cues)
        self.cue_act: DefaultDict[str, CounterType[str]] = defaultdict(Counter)
        self.act_totals: CounterType[str] = Counter()
        self.cue_touches: CounterType[str] = Counter()
        self.observations = 0

    @staticmethod
    def _replace_concepts(utterance: str, concepts: Sequence[str]) -> str:
        text = utterance.lower()
        for idx, concept in sorted(enumerate(concepts), key=lambda x: len(x[1]), reverse=True):
            c = normalize_name(concept)
            if c:
                text = re.sub(rf"\b{re.escape(c)}\b", f" conceptslot{idx} ", text, flags=re.I)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _cues(text: str, max_n: int = 4) -> List[str]:
        toks = tokenize(text)
        rows = set()
        for n in range(1, min(max_n, len(toks)) + 1):
            for i in range(len(toks)-n+1):
                phrase = " ".join(toks[i:i+n])
                rows.add("A:" + phrase)
                if i == 0:
                    rows.add("S:" + phrase)
                if i+n == len(toks):
                    rows.add("E:" + phrase)
        return sorted(rows)

    def observe(self, episode: DialogueEpisode) -> None:
        text = self._replace_concepts(episode.utterance, episode.concepts)
        self.act_totals[episode.act] += 1
        for cue in self._cues(text):
            self.cue_act[cue][episode.act] += 1
            self.cue_touches[cue] += 1
        self.observations += 1
        self._prune()

    def _prune(self) -> None:
        if len(self.cue_act) <= self.max_cues:
            return
        ranked = sorted(self.cue_act, key=lambda c: (self.cue_touches[c], sum(self.cue_act[c].values()), c))
        for cue in ranked[:len(self.cue_act)-self.max_cues]:
            self.cue_act.pop(cue, None); self.cue_touches.pop(cue, None)

    def predict(self, utterance: str, concepts: Sequence[str] = ()) -> Tuple[Optional[str], float, List[Tuple[str,float]]]:
        text = self._replace_concepts(utterance, concepts)
        totals = max(1, sum(self.act_totals.values()))
        scores: Dict[str, float] = defaultdict(float)
        evidence: DefaultDict[str, List[Tuple[float,str]]] = defaultdict(list)
        for cue in self._cues(text):
            row = self.cue_act.get(cue)
            if not row:
                continue
            support = sum(row.values())
            if support < 3:
                continue
            phrase = cue[2:]
            n = len(phrase.split())
            position_bonus = 1.10 if cue.startswith(("S:", "E:")) else 1.0
            for act, count in row.items():
                purity = count/support
                baseline = self.act_totals.get(act, 0)/totals
                discrimination = purity - baseline
                if discrimination <= .035:
                    continue
                weight = discrimination * math.log1p(count) * (1.0 + .10*(n-1)) * position_bonus
                scores[act] += weight
                evidence[act].append((weight, cue))
        if not scores:
            return None, 0.0, []
        ordered = sorted(((score, act) for act, score in scores.items()), reverse=True)
        top, act = ordered[0]
        second = ordered[1][0] if len(ordered) > 1 else 0.0
        ratio = top/max(top+second, 1e-9)
        support_factor = 1.0 - math.exp(-self.act_totals.get(act, 0)/10.0)
        confidence = float(ratio*(.68+.32*support_factor))
        ev = sorted(evidence[act], reverse=True)[:5]
        return act, confidence, [(cue, float(weight)) for weight, cue in ev]

    def summary(self, limit: int = 20) -> Dict[str, object]:
        rows = []
        for cue, counter in self.cue_act.items():
            total = sum(counter.values())
            if total < 3:
                continue
            act, count = counter.most_common(1)[0]
            rows.append((count/total, total, cue, act))
        rows.sort(reverse=True)
        return {
            "version": self.VERSION,
            "observations": self.observations,
            "acts": dict(self.act_totals),
            "cues": len(self.cue_act),
            "max_cues": self.max_cues,
            "strongest": [
                {"cue": cue, "act": act, "purity": purity, "support": support}
                for purity, support, cue, act in rows[:limit]
            ],
            "raw_sentences_retained": 0,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "max_cues": self.max_cues,
            "observations": self.observations,
            "act_totals": dict(self.act_totals),
            "cue_touches": dict(self.cue_touches),
            "cue_act": {k: dict(v) for k,v in self.cue_act.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "DialogueActLearner":
        obj = cls(int(data.get("max_cues", 24000)))
        obj.observations = int(data.get("observations", 0))
        obj.act_totals.update({str(k):int(v) for k,v in dict(data.get("act_totals",{})).items()})
        obj.cue_touches.update({str(k):int(v) for k,v in dict(data.get("cue_touches",{})).items()})
        for cue,row in dict(data.get("cue_act",{})).items():
            obj.cue_act[str(cue)].update({str(k):int(v) for k,v in dict(row).items()})
        obj._prune(); return obj

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DialogueActLearner":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def train_dialogue_learner(learner: DialogueActLearner, teacher: ConversationTeacherV15, steps: int) -> Dict[str, object]:
    steps = max(1, int(steps)); correct = 0
    for _ in range(steps):
        ep = teacher.episode(held_out=False)
        pred, _, _ = learner.predict(ep.utterance, ep.concepts)
        correct += int(pred == ep.act)
        learner.observe(ep)
    return {
        "steps": steps,
        "correct_before_learning": correct/steps,
        "summary": learner.summary(12),
    }


def test_dialogue_learner(learner: DialogueActLearner, teacher: ConversationTeacherV15, samples: int = 240) -> Dict[str, object]:
    samples = max(1, int(samples)); correct = 0; by_act: Dict[str,List[int]] = defaultdict(lambda:[0,0]); failures=[]
    before = learner.observations
    for _ in range(samples):
        ep = teacher.episode(held_out=True)
        pred, conf, evidence = learner.predict(ep.utterance, ep.concepts)
        ok = pred == ep.act
        correct += int(ok); by_act[ep.act][0] += int(ok); by_act[ep.act][1] += 1
        if not ok and len(failures) < 20:
            failures.append({"utterance":ep.utterance,"expected":ep.act,"predicted":pred,"confidence":conf,"evidence":evidence})
    return {
        "samples": samples,
        "accuracy": correct/samples,
        "by_act": {k:(v[0]/v[1] if v[1] else 0.0) for k,v in by_act.items()},
        "memory_frozen": learner.observations == before,
        "failures": failures,
    }
