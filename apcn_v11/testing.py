from __future__ import annotations

from collections import Counter
from typing import Optional

from apcn_v10.language_session import GeneratedLanguageTestReport, LanguageFailure, AdaptiveLanguageSession
from apcn_v10.semantic import SemanticNode

from .discourse import DiscourseEntityRegistry
from .language_teacher import RichSemanticTeacherV11


def semantic_equal_instances(a: Optional[SemanticNode], b: Optional[SemanticNode]) -> bool:
    if a is None or b is None:
        return False
    if a.op != b.op or a.relation != b.relation:
        return False
    if (a.subject is None) != (b.subject is None) or (a.object is None) != (b.object is None):
        return False
    if a.subject is not None and b.subject is not None:
        if (a.subject.color, a.subject.shape, a.subject.instance) != (b.subject.color, b.subject.shape, b.subject.instance):
            return False
    if a.object is not None and b.object is not None:
        if (a.object.color, a.object.shape, a.object.instance) != (b.object.color, b.object.shape, b.object.instance):
            return False
    if len(a.children) != len(b.children):
        return False
    return all(semantic_equal_instances(x, y) for x, y in zip(a.children, b.children))


def run_generated_language_test_v11(learner, samples: int = 600, seed: int = 11011,
                                    keep_failures: int = 40) -> GeneratedLanguageTestReport:
    """Held-out V0.11 benchmark with discourse identity tested explicitly.

    For reference episodes the learner first interprets the preceding sentence,
    builds its own discourse registry from that prediction, and then resolves the
    continuation. Teacher entity IDs are never injected into the registry.
    """
    teacher = RichSemanticTeacherV11(seed)
    skills = list(AdaptiveLanguageSession.SKILLS)
    intents = ["ASSERT", "QUERY", "GOAL"]
    relations = list(teacher.relations)
    iidx = {v: i for i, v in enumerate(intents)}
    ridx = {v: i for i, v in enumerate(relations)}
    imat = [[0 for _ in intents] for _ in intents]
    rmat = [[0 for _ in relations] for _ in relations]
    before = learner.episode_count
    exact = intent_ok = relation_ok = operator_ok = 0
    skill_hits = Counter()
    skill_totals = Counter()
    failures = []

    for idx in range(max(1, int(samples))):
        skill = skills[idx % len(skills)]
        episodes = teacher.for_skill(skill, held_out=True)
        registry = DiscourseEntityRegistry()
        if skill == "reference":
            first = episodes[0]
            first_pred = learner.parse(first.utterance, discourse_registry=registry)
            registry.ingest(first_pred)
            ep = episodes[1]
            pred = learner.parse(ep.utterance, discourse_registry=registry)
        else:
            ep = episodes[-1]
            pred = learner.parse(ep.utterance)

        ex = semantic_equal_instances(pred, ep.program)
        exact += int(ex)
        skill_totals[skill] += 1
        skill_hits[skill] += int(ex)

        truth_intent = ep.program.intent()
        pred_intent = pred.intent() if pred else None
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
        raise RuntimeError("V0.11 generated language test modified learner memory")
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
