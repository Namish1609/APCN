from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

from .dialogue_learning import train_dialogue_learner
from .session import CognitiveSessionV15


@dataclass
class ConversationBenchmarkReport:
    turns: int
    act_accuracy: float
    required_content_accuracy: float
    unknown_honesty: float
    learned_memory_accuracy: float
    followup_accuracy: float
    dev_dialogue_act_accuracy: float
    knowledge_language_separation: float
    visual_experiences_changed: int
    benchmark_role: str
    failures: List[Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _seed_science(s: CognitiveSessionV15) -> None:
    for name in ("distance", "time", "velocity change", "mass", "volume"):
        s.concepts.add_primitive(name, grounded=True)
    for sentence in (
        "speed is distance divided by time",
        "acceleration is velocity change divided by time",
        "density is mass divided by volume",
    ):
        s.concepts.learn_definition(sentence)


def _grade_script(s: CognitiveSessionV15, script, failures, *, label: str):
    act_ok = content_ok = 0
    for prompt, expected_act, required in script:
        reply = s.talk(prompt)
        aok = reply.act == expected_act
        lower = reply.text.lower()
        cok = all(term.lower() in lower for term in required)
        act_ok += int(aok)
        content_ok += int(cok)
        if not (aok and cok) and len(failures) < 30:
            failures.append({
                "suite": label,
                "prompt": prompt,
                "expected_act": expected_act,
                "actual_act": reply.act,
                "required": required,
                "reply": reply.text,
                "confidence": reply.confidence,
                "trace": reply.trace,
            })
    n = max(1, len(script))
    return act_ok / n, content_ok / n


def _architecture_separation_check(seed: int) -> float:
    """Dialogue exposure must not manufacture concept/world knowledge."""
    s = CognitiveSessionV15(seed)
    checks = []

    # Language cue learning about definition requests must not create the target
    # concept itself.
    checks.append(not bool(s.concepts.understanding("glorpnax").get("known")))
    train_dialogue_learner(s.dialogue_learner_v15, s.dialogue_teacher_v15, 1200)
    checks.append(not bool(s.concepts.understanding("glorpnax").get("known")))

    # Conversely, concept knowledge must remain present independently of the
    # dialogue router's accumulated statistics.
    _seed_science(s)
    checks.append(bool(s.concepts.understanding("acceleration").get("known")))
    router_observations = s.dialogue_learner_v15.observations
    s.dialogue_learner_v15.cue_act.clear()
    s.dialogue_learner_v15.cue_touches.clear()
    s.dialogue_learner_v15.act_totals.clear()
    s.dialogue_learner_v15.observations = 0
    checks.append(bool(s.concepts.understanding("acceleration").get("known")))
    checks.append(router_observations > 0)

    audit = s.conversation_memory_audit()["architecture_contract"]
    checks.append(bool(audit["language_is_semantic_compiler"]))
    checks.append(bool(audit["concept_world_memory_is_knowledge_substrate"]))
    return sum(bool(x) for x in checks) / len(checks)


def run_conversation_benchmark(seed: int = 1501) -> ConversationBenchmarkReport:
    s = CognitiveSessionV15(seed)
    _seed_science(s)
    visual_before = s.visual.learner.episode_count

    # Smoke conversation verifies executable semantics, dialogue context and
    # explicit teachability. It is intentionally not presented as a blind English
    # benchmark.
    script = [
        ("hello", "GREETING", ("ready",)),
        ("what is acceleration?", "ANSWER_DEFINITION", ("acceleration", "velocity change", "time")),
        ("what does it depend on?", "ANSWER_DEPENDENCIES", ("velocity change", "time")),
        ("why?", "ANSWER_PROVENANCE", ("acceleration",)),
        ("tell me more", "ANSWER_MORE", ("dependency",)),
        ("fluxion means acceleration", "LEARN_ALIAS", ("fluxion", "acceleration")),
        ("what is fluxion?", "ANSWER_DEFINITION", ("acceleration",)),
        ("remember that orbix is a sensor", "LEARN_FACT", ("orbix", "sensor")),
        ("what is orbix?", "ANSWER_FACT", ("orbix", "sensor")),
        ("is orbix a sensor?", "ANSWER_FACT", ("yes", "orbix", "sensor")),
        ("what did I just teach you?", "ANSWER_MEMORY", ("orbix",)),
        ("compare speed and density", "ANSWER_COMPARE", ("speed", "density")),
        ("do you understand density?", "ANSWER_KNOWLEDGE", ("density",)),
        ("do you understand it?", "ANSWER_KNOWLEDGE", ("density",)),
        ("what are we talking about?", "ANSWER_MEMORY", ("density",)),
        ("what is glorpnax?", "CLARIFY", ("do not currently",)),
        ("please frobnicate the ontology sideways", "CLARIFY", ("do not yet know",)),
    ]

    failures: List[Dict[str, object]] = []
    act_accuracy, content_accuracy = _grade_script(s, script, failures, label="smoke")

    # The old TEST templates are now explicitly development-only because their
    # failures were already inspected during V0.15 development. They can measure
    # progress, but they are not evidence of blind generalization.
    dialogue_report = s.test_dialogue_generalization(360)

    unknown_total = unknown_ok = learned_total = learned_ok = follow_total = follow_ok = 0
    s2 = CognitiveSessionV15(seed + 1)
    _seed_science(s2)
    follow_prompts = {"what does it depend on?", "why?", "tell me more", "do you understand it?", "what are we talking about?"}
    for prompt, expected_act, required in script:
        reply = s2.talk(prompt)
        lower = reply.text.lower()
        ok = reply.act == expected_act and all(x.lower() in lower for x in required)
        if expected_act == "CLARIFY":
            unknown_total += 1
            unknown_ok += int(reply.act == "CLARIFY" and reply.confidence < .5)
        if expected_act.startswith("LEARN_"):
            learned_total += 1
            learned_ok += int(reply.learned)
        if prompt.lower() in follow_prompts:
            follow_total += 1
            follow_ok += int(ok)

    visual_after = s.visual.learner.episode_count
    separation = _architecture_separation_check(seed + 2)
    return ConversationBenchmarkReport(
        turns=len(script),
        act_accuracy=act_accuracy,
        required_content_accuracy=content_accuracy,
        unknown_honesty=unknown_ok / max(1, unknown_total),
        learned_memory_accuracy=learned_ok / max(1, learned_total),
        followup_accuracy=follow_ok / max(1, follow_total),
        dev_dialogue_act_accuracy=float(dialogue_report["accuracy"]),
        knowledge_language_separation=float(separation),
        visual_experiences_changed=visual_after - visual_before,
        benchmark_role="development_and_architecture_contract_not_blind_final",
        failures=failures + [{"suite": "dev_dialogue_classifier", **x} for x in dialogue_report["failures"][:10]],
    )
