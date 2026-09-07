from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

from .semantics import SemanticClause
from .session import CognitiveSessionV17


@dataclass
class StructuredBenchmarkReport:
    operator_parse_accuracy: float
    nested_roundtrip_exact: float
    generation_min_variants: int
    discourse_reference_accuracy: float
    universal_inference_accuracy: float
    conditional_reasoning_accuracy: float
    causal_retrieval_accuracy: float
    temporal_order_accuracy: float
    negation_accuracy: float
    unknown_honesty: float
    truth_firewall: float
    raw_chat_persisted: bool
    visual_experiences_changed: int
    benchmark_role: str
    failures: List[Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _atom(subject: str, event: str, tense: str = "present") -> SemanticClause:
    return SemanticClause.make("EVENT", subject=subject, event=event, tense=tense)


def run_structured_benchmark(seed: int = 17001) -> StructuredBenchmarkReport:
    s = CognitiveSessionV17(seed)
    visual_before = s.visual.learner.episode_count
    failures: List[Dict[str, object]] = []

    structures = [
        ("not milo is active", SemanticClause.make("NOT", children=(SemanticClause.make("PROPERTY", subject="milo", property="active", tense="present"),))),
        ("lamp turns off because power fails", SemanticClause.make("CAUSE", children=(_atom("power", "fail"), _atom("lamp", "turn off")))),
        ("if battery is empty then device stops", SemanticClause.make("IF", children=(SemanticClause.make("PROPERTY", subject="battery", property="empty", tense="present"), _atom("device", "stop")))),
        ("door opens before light turns on", SemanticClause.make("BEFORE", children=(_atom("door", "open"), _atom("light", "turn on")))),
        ("every cat is a creature", SemanticClause.make("FORALL_ISA", kind="cat", category="creature")),
        ("some sensor is active", SemanticClause.make("EXISTS_PROPERTY", kind="sensor", property="active")),
        ("both milo is active and tilo is quiet", SemanticClause.make("AND", children=(SemanticClause.make("PROPERTY", subject="milo", property="active", tense="present"), SemanticClause.make("PROPERTY", subject="tilo", property="quiet", tense="present")))),
    ]

    parse_ok = 0
    variant_counts: List[int] = []
    roundtrip_total = roundtrip_ok = 0
    for surface, expected in structures:
        parsed, _, evidence = s.structured_language_v17.parse(surface)
        ok = parsed == expected
        parse_ok += int(ok)
        if not ok:
            failures.append({"suite": "parse", "surface": surface, "expected": expected.to_dict(), "parsed": None if parsed is None else parsed.to_dict(), "evidence": evidence})
        rep = s.structured_language_v17.roundtrip(expected, 12)
        variant_counts.append(int(rep["count"]))
        for row in rep["generations"]:
            roundtrip_total += 1
            roundtrip_ok += int(row["exact"])
            if not row["exact"] and len(failures) < 40:
                failures.append({"suite": "roundtrip", "semantic": expected.to_dict(), "row": row})

    # Nested operator composition that was not a single bootstrap demonstration.
    nested = SemanticClause.make(
        "IF",
        children=(
            SemanticClause.make("NOT", children=(SemanticClause.make("PROPERTY", subject="battery", property="charged", tense="present"),)),
            _atom("device", "stop"),
        ),
    )
    nested_rep = s.structured_language_v17.roundtrip(nested, 12)
    for row in nested_rep["generations"]:
        roundtrip_total += 1
        roundtrip_ok += int(row["exact"])

    # Discourse reference: no raw transcript is retained; `it` resolves through
    # bounded semantic focus.
    s.talk("remember that milo is a cat")
    s.talk("remember that it is active")
    discourse = s.talk("is it true that it is active?")
    discourse_ok = int(discourse.act == "ANSWER_TRUE" and s.discourse_v17.focus == "milo")

    s.talk("remember that every cat is a creature")
    universal = s.talk("is it true that milo is a creature?")
    universal_ok = int(universal.act == "ANSWER_TRUE" and any("universal_inference" in x for x in universal.trace))

    s.talk("remember that if battery is empty then device stops")
    conditional = s.talk("what follows if battery is empty?")
    conditional_ok = int(conditional.act == "ANSWER_CONSEQUENCE" and "device stops" in conditional.text.lower())

    s.talk("remember that lamp turns off because power fails")
    causal = s.talk("what causes lamp turns off?")
    causal_ok = int(causal.act == "ANSWER_CAUSE" and "power fails" in causal.text.lower())

    s.talk("remember that door opens before light turns on")
    temporal = s.talk("what happens before light turns on?")
    temporal_ok = int(temporal.act == "ANSWER_BEFORE" and "door opens" in temporal.text.lower())

    s.talk("remember that not sensor is active")
    negated = s.talk("is it true that sensor is active?")
    negation_ok = int(negated.act == "ANSWER_FALSE")

    unknown = s.talk("is it true that zorbin is active?")
    unknown_ok = int(unknown.act == "ANSWER_UNKNOWN" and "zorbin" in unknown.text.lower())

    truth_firewall = int(
        not hasattr(s.structured_language_v17, "semantic_memory")
        and not hasattr(s.structured_language_v17, "world")
        and not hasattr(s.structured_language_v17, "concepts")
        and not hasattr(s.operator_memory_v17, "world")
    )

    visual_after = s.visual.learner.episode_count
    return StructuredBenchmarkReport(
        operator_parse_accuracy=parse_ok / max(1, len(structures)),
        nested_roundtrip_exact=roundtrip_ok / max(1, roundtrip_total),
        generation_min_variants=min(variant_counts[1:]) if len(variant_counts) > 1 else 0,
        discourse_reference_accuracy=float(discourse_ok),
        universal_inference_accuracy=float(universal_ok),
        conditional_reasoning_accuracy=float(conditional_ok),
        causal_retrieval_accuracy=float(causal_ok),
        temporal_order_accuracy=float(temporal_ok),
        negation_accuracy=float(negation_ok),
        unknown_honesty=float(unknown_ok),
        truth_firewall=float(truth_firewall),
        raw_chat_persisted=bool(s.discourse_v17.summary()["raw_chat_transcript_persisted"]),
        visual_experiences_changed=visual_after - visual_before,
        benchmark_role="controlled_architecture_and_composition_gate_not_open_english",
        failures=failures,
    )
