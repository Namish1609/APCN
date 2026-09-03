from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import json
import math
import random

from .semantic import EntityRef, SemanticNode, semantic_equal
from .language_common import LanguageEpisode, SkillState
from .language_teacher import RichSemanticTeacher
from .language_learner import SemanticLanguageLearnerV10

@dataclass
class AdaptiveStep:
    skill: str
    episode: LanguageEpisode
    prediction: Optional[SemanticNode]
    correct_before_learning: bool


class AdaptiveLanguageSession:
    SKILLS = ("grounding", "intent", "group", "sequence", "negation", "reference")

    def __init__(self, seed: int = 10, learner: Optional[SemanticLanguageLearnerV10] = None):
        self.seed = seed
        self.teacher = RichSemanticTeacher(seed)
        self.learner = learner or SemanticLanguageLearnerV10()
        self.rng = random.Random(seed + 99)
        self.skills: Dict[str, SkillState] = {k: SkillState() for k in self.SKILLS}
        self.history: List[Dict[str, float]] = []
        self.last_skill = "grounding"

    def _unlocked(self) -> List[str]:
        n = self.learner.episode_count
        s = self.skills
        out = ["grounding"]
        if n >= 100 or s["grounding"].ema >= 0.45:
            out.append("intent")
        if n >= 280 or s["intent"].ema >= 0.52:
            out.append("group")
        if n >= 500 or s["group"].ema >= 0.50:
            out.extend(("sequence", "negation"))
        if n >= 800 or min(s["sequence"].ema, s["negation"].ema) >= 0.48:
            out.append("reference")
        return out

    def choose_skill(self) -> str:
        unlocked = self._unlocked()
        minimum = min(self.skills[k].attempts for k in unlocked)
        eligible = [k for k in unlocked if self.skills[k].attempts <= minimum + 36]
        scored = []
        for skill in eligible:
            st = self.skills[skill]
            uncertainty = 1.0 - st.ema
            exploration = 0.24 / math.sqrt(st.attempts + 1.0)
            repeat_penalty = 0.06 if skill == self.last_skill else 0.0
            scored.append((uncertainty + exploration - repeat_penalty + self.rng.random()*0.025, skill))
        return max(scored)[1]

    @staticmethod
    def _skill_correct(skill: str, pred: Optional[SemanticNode], truth: SemanticNode) -> bool:
        if pred is None:
            return False
        if skill in {"grounding", "reference"}:
            return semantic_equal(pred, truth)
        if skill == "intent":
            return pred.intent() == truth.intent() and pred.relations()[:1] == truth.relations()[:1]
        if skill in {"group", "sequence", "negation"}:
            opname = {"group": "GROUP", "sequence": "SEQUENCE", "negation": "NEGATE"}[skill]
            return opname in pred.operators() and pred.relations() == truth.relations()
        return semantic_equal(pred, truth)

    def step(self) -> AdaptiveStep:
        skill = self.choose_skill()
        episodes = self.teacher.for_skill(skill, held_out=False)
        result: Optional[AdaptiveStep] = None
        discourse_focus: Optional[EntityRef] = None
        for ep in episodes:
            focus = ep.discourse_focus or discourse_focus
            pred = self.learner.parse(ep.utterance, focus)
            ok = self._skill_correct(skill, pred, ep.program)
            self.skills[skill].update(ok)
            self.learner.observe(ep)
            atom = ep.program.atom()
            if atom is not None and atom.subject is not None:
                discourse_focus = atom.subject
            result = AdaptiveStep(skill, ep, pred, ok)
        self.last_skill = skill
        assert result is not None
        return result

    def competence(self) -> Dict[str, float]:
        return {k: self.skills[k].ema for k in self.SKILLS}

    def save(self, output_dir: str | Path = "outputs/v0_10") -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        memory = out / "language_memory_v0_10.json"
        self.learner.save(memory)
        (out / "language_session_v0_10.json").write_text(json.dumps({
            "version": "0.10.0",
            "seed": self.seed,
            "skills": {k: asdict(v) for k, v in self.skills.items()},
            "history": self.history,
        }, indent=2), encoding="utf-8")
        return memory


@dataclass
class LanguageFailure:
    skill: str
    utterance: str
    expected: str
    predicted: str


@dataclass
class GeneratedLanguageTestReport:
    samples: int
    exact_accuracy: float
    intent_accuracy: float
    relation_accuracy: float
    operator_accuracy: float
    skill_accuracy: Dict[str, float]
    intent_labels: List[str]
    intent_confusion: List[List[int]]
    relation_labels: List[str]
    relation_confusion: List[List[int]]
    failures: List[LanguageFailure]
    learner_episode_count_before: int
    learner_episode_count_after: int


def run_generated_language_test(
    learner: SemanticLanguageLearnerV10,
    samples: int = 600,
    seed: int = 10010,
    keep_failures: int = 40,
) -> GeneratedLanguageTestReport:
    """Balanced, generated, held-out-template test. Never updates memory."""
    teacher = RichSemanticTeacher(seed)
    skills = list(AdaptiveLanguageSession.SKILLS)
    intents = ["ASSERT", "QUERY", "GOAL"]
    relations = list(teacher.relations)
    iidx = {v:i for i,v in enumerate(intents)}
    ridx = {v:i for i,v in enumerate(relations)}
    imat = [[0 for _ in intents] for _ in intents]
    rmat = [[0 for _ in relations] for _ in relations]
    before = learner.episode_count
    exact = intent_ok = relation_ok = operator_ok = 0
    skill_hits = Counter()
    skill_totals = Counter()
    failures: List[LanguageFailure] = []
    discourse_focus: Optional[EntityRef] = None

    for idx in range(max(1, int(samples))):
        skill = skills[idx % len(skills)]
        episodes = teacher.for_skill(skill, held_out=True)
        ep = episodes[-1]
        if skill == "reference":
            first = episodes[0]
            first_atom = first.program.atom()
            discourse_focus = first_atom.subject if first_atom is not None else None
        else:
            discourse_focus = ep.discourse_focus
        pred = learner.parse(ep.utterance, discourse_focus)
        ex = semantic_equal(pred, ep.program)
        exact += int(ex)
        skill_totals[skill] += 1
        skill_hits[skill] += int(ex)

        truth_intent, pred_intent = ep.program.intent(), pred.intent() if pred else None
        if truth_intent is not None:
            intent_ok += int(pred_intent == truth_intent)
            if truth_intent in iidx and pred_intent in iidx:
                imat[iidx[truth_intent]][iidx[pred_intent]] += 1
        truth_rel = ep.program.relations()[0] if ep.program.relations() else None
        pred_rel = pred.relations()[0] if pred and pred.relations() else None
        if truth_rel is not None:
            relation_ok += int(pred_rel == truth_rel)
            if truth_rel in ridx and pred_rel in ridx:
                rmat[ridx[truth_rel]][ridx[pred_rel]] += 1
        truth_ops = set(ep.program.operators())
        pred_ops = set(pred.operators()) if pred else set()
        operator_ok += int(truth_ops == pred_ops)
        if not ex and len(failures) < keep_failures:
            failures.append(LanguageFailure(
                skill=skill,
                utterance=ep.utterance,
                expected=ep.program.pretty(),
                predicted="NO PARSE" if pred is None else pred.pretty(),
            ))

    after = learner.episode_count
    if after != before:
        raise RuntimeError("generated language test modified learner memory")
    n = max(1, int(samples))
    return GeneratedLanguageTestReport(
        samples=n,
        exact_accuracy=exact/n,
        intent_accuracy=intent_ok/n,
        relation_accuracy=relation_ok/n,
        operator_accuracy=operator_ok/n,
        skill_accuracy={k: skill_hits[k]/max(1, skill_totals[k]) for k in skills},
        intent_labels=intents,
        intent_confusion=imat,
        relation_labels=relations,
        relation_confusion=rmat,
        failures=failures,
        learner_episode_count_before=before,
        learner_episode_count_after=after,
    )
