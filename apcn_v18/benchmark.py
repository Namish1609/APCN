from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

from .realizer import AnswerPlanV18
from .session import CognitiveSessionV18


@dataclass
class NaturalConversationBenchmarkV18:
    immediate_alias_accuracy: float
    lexical_repair_accuracy: float
    ordinary_property_truth_accuracy: float
    ordinary_isa_truth_accuracy: float
    about_aggregation_accuracy: float
    universal_inference_accuracy: float
    proof_followup_accuracy: float
    conditional_accuracy: float
    causal_accuracy: float
    temporal_accuracy: float
    unknown_honesty: float
    property_category_separation: float
    natural_realization_min_variants: int
    content_firewall: float
    visual_experiences_changed: int
    benchmark_role: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _seed_definition_knowledge(s: CognitiveSessionV18) -> None:
    for name in ("distance", "time", "velocity change"):
        s.concepts.add_primitive(name, grounded=True)
    s.concepts.learn_definition("acceleration is velocity change divided by time")
    s.v18_binding_repairs = s._repair_knowledge_bindings()
    s._rebuild_v18_language()


def run_natural_conversation_benchmark(seed: int = 18012) -> NaturalConversationBenchmarkV18:
    s = CognitiveSessionV18(seed)
    _seed_definition_knowledge(s)
    visual_before = s.visual.learner.episode_count

    s.talk("fluxion means acceleration")
    alias = s.talk("what is fluxion?")
    immediate_alias = int(alias.act == "ANSWER_DEFINITION" and "velocity change" in alias.text.lower() and "time" in alias.text.lower())

    repair = s.talk("I asked what is Fluxation")
    lexical_repair = int(repair.act == "CLARIFY" and "fluxion" in repair.text.lower() and "did you mean" in repair.text.lower())

    s.talk("remember that milo is a cat")
    s.talk("remember that milo is active")
    prop = s.talk("is milo active?")
    isa = s.talk("is milo a cat?")
    ordinary_prop = int(prop.act == "ANSWER_TRUE" and "milo" in prop.text.lower() and "active" in prop.text.lower())
    ordinary_isa = int(isa.act == "ANSWER_TRUE" and "cat" in isa.text.lower())

    about = s.talk("what do you know about milo?")
    about_ok = int(about.act == "ANSWER_ABOUT" and "cat" in about.text.lower() and "active" in about.text.lower())

    s.talk("remember that every cat is a creature")
    inferred = s.talk("is milo a creature?")
    universal = int(inferred.act == "ANSWER_TRUE" and "creature" in inferred.text.lower() and any("inference" in str(x) for x in inferred.trace))
    why = s.talk("why?")
    proof = int("cat" in why.text.lower() and "creature" in why.text.lower() and any("last_proof" in str(x) for x in why.trace))

    s.talk("remember that if battery is empty then device stops")
    follows = s.talk("what follows if battery is empty?")
    conditional = int("device stops" in follows.text.lower())

    s.talk("remember that power fails causes lamp turns off")
    cause = s.talk("what causes lamp turns off?")
    causal = int(cause.act == "ANSWER_CAUSE" and "power fails" in cause.text.lower())

    s.talk("remember that door opens before light turns on")
    before = s.talk("what happens before light turns on?")
    temporal = int(before.act == "ANSWER_BEFORE" and "door opens" in before.text.lower())

    unknown = s.talk("is zorbin a creature?")
    unknown_ok = int(unknown.act == "ANSWER_UNKNOWN" and "zorbin" in unknown.text.lower() and "don't know" in unknown.text.lower())

    milo_categories = {rec.object for rec in s.facts_v15.about("milo") if rec.relation == "is_a"}
    category_separation = int("cat" in milo_categories and "active" not in milo_categories)

    plan = AnswerPlanV18.make("ANSWER_TRUE_DERIVED", statement="milo is a creature", reason="milo is a cat, and every cat is a creature")
    variants = s.generate_answer_variants(plan, 12)["surfaces"]

    realizer = s.natural_realizer_v18
    firewall = int(not hasattr(realizer, "semantic_memory") and not hasattr(realizer, "concepts") and not hasattr(realizer, "world") and not hasattr(realizer.memory, "semantic_memory") and not hasattr(realizer.memory, "concepts") and not hasattr(realizer.memory, "world"))

    visual_after = s.visual.learner.episode_count
    return NaturalConversationBenchmarkV18(
        immediate_alias_accuracy=float(immediate_alias),
        lexical_repair_accuracy=float(lexical_repair),
        ordinary_property_truth_accuracy=float(ordinary_prop),
        ordinary_isa_truth_accuracy=float(ordinary_isa),
        about_aggregation_accuracy=float(about_ok),
        universal_inference_accuracy=float(universal),
        proof_followup_accuracy=float(proof),
        conditional_accuracy=float(conditional),
        causal_accuracy=float(causal),
        temporal_accuracy=float(temporal),
        unknown_honesty=float(unknown_ok),
        property_category_separation=float(category_separation),
        natural_realization_min_variants=len(variants),
        content_firewall=float(firewall),
        visual_experiences_changed=visual_after - visual_before,
        benchmark_role="desktop_transcript_regression_and_architecture_gate_not_blind_general_english",
    )
