from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
import random

from .learner import SemanticLanguageLearner
from .semantic import SemanticNode, semantic_equal
from .teacher import SemanticTeacher


@dataclass
class SemanticFailure:
    utterance: str
    expected: str
    predicted: str
    kind: str


@dataclass
class SemanticTestReport:
    samples: int
    exact_accuracy: float
    intent_accuracy: float
    relation_accuracy: float
    operator_accuracy: float
    confusion_labels: List[str]
    intent_confusion: List[List[int]]
    relation_labels: List[str]
    relation_confusion: List[List[int]]
    failures: List[SemanticFailure]
    learner_episode_count_before: int
    learner_episode_count_after: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _primary_relation(p: Optional[SemanticNode]) -> Optional[str]:
    if p is None:
        return None
    rels = p.relations()
    return rels[0] if rels else None


def _primary_operator(p: Optional[SemanticNode]) -> str:
    if p is None:
        return "NONE"
    ops = p.operators()
    return ops[0] if ops else "NONE"


def run_semantic_test(learner: SemanticLanguageLearner, samples: int = 300, seed: int = 9090, held_out_templates: bool = False, keep_failures: int = 40) -> SemanticTestReport:
    teacher = SemanticTeacher(seed=seed)
    before = learner.episode_count
    intents = ["ASSERT", "QUERY", "GOAL"]
    relations = list(teacher.relations)
    iidx = {x: i for i, x in enumerate(intents)}
    ridx = {x: i for i, x in enumerate(relations)}
    imat = [[0] * len(intents) for _ in intents]
    rmat = [[0] * len(relations) for _ in relations]
    exact = intent_ok = relation_ok = operator_ok = 0
    failures: List[SemanticFailure] = []

    for i in range(samples):
        mode = i % 10
        if mode == 0:
            ep = teacher.group()
        elif mode == 1:
            ep = teacher.negated()
        elif mode == 2:
            ep = teacher.sequence()
        else:
            ep = teacher.simple(intent=intents[i % 3], relation=relations[(i // 3) % len(relations)], held_out=held_out_templates)
        pred = learner.parse(ep.utterance, ep.discourse_focus)
        truth = ep.program
        ti, pi = truth.intent() or "ASSERT", pred.intent() if pred else None
        tr, pr = _primary_relation(truth), _primary_relation(pred)
        if ti in iidx and pi in iidx:
            imat[iidx[ti]][iidx[pi]] += 1
        if tr in ridx and pr in ridx:
            rmat[ridx[tr]][ridx[pr]] += 1
        eok = semantic_equal(pred, truth)
        iok, rok = pi == ti, pr == tr
        ook = _primary_operator(pred) == _primary_operator(truth)
        exact += int(eok); intent_ok += int(iok); relation_ok += int(rok); operator_ok += int(ook)
        if not eok and len(failures) < keep_failures:
            kind = "operator" if not ook else "intent" if not iok else "relation" if not rok else "arguments"
            failures.append(SemanticFailure(ep.utterance, truth.pretty(), "<parse failed>" if pred is None else pred.pretty(), kind))

    after = learner.episode_count
    if after != before:
        raise RuntimeError("semantic test changed learner memory")
    return SemanticTestReport(samples, exact/samples, intent_ok/samples, relation_ok/samples, operator_ok/samples, intents, imat, relations, rmat, failures, before, after)
