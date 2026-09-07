from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

from .bidirectional import SemanticFrame
from .compositional import RECOMBINATION_SPLIT
from .session import CognitiveSessionV16


@dataclass
class BidirectionalBenchmarkReport:
    request_accuracy: float
    generation_min_variants: int
    generation_mean_variants: float
    roundtrip_exact: float
    unknown_honesty: float
    explicit_learning_transfer: float
    content_firewall: float
    visual_experiences_changed: int
    recombination_accuracy: float
    dev_parse_accuracy: float
    benchmark_role: str
    failures: List[Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _seed(s: CognitiveSessionV16) -> None:
    for name in ("distance", "time", "velocity change", "mass", "volume"):
        s.concepts.add_primitive(name, grounded=True)
    s.concepts.learn_definition("speed is distance divided by time")
    s.concepts.learn_definition("acceleration is velocity change divided by time")
    s.concepts.learn_definition("density is mass divided by volume")
    s._rebuild_v16_language()


def run_bidirectional_benchmark(seed: int = 16001) -> BidirectionalBenchmarkReport:
    s = CognitiveSessionV16(seed)
    _seed(s)
    visual_before = s.visual.learner.episode_count
    failures: List[Dict[str, object]] = []

    requests = [
        ("what is acceleration", "ANSWER_DEFINITION", ("acceleration", "velocity change", "time")),
        ("what does speed depend on", "ANSWER_DEPENDENCIES", ("speed", "distance", "time")),
        ("do you know density", "ANSWER_KNOWLEDGE", ("density",)),
        ("compare speed and density", "ANSWER_COMPARE", ("speed", "density")),
    ]
    req_ok = 0
    for prompt, act, required in requests:
        row = s.talk(prompt)
        ok = row.act == act and all(x in row.text.lower() for x in required) and any("v016:realize" in x for x in row.trace)
        req_ok += int(ok)
        if not ok:
            failures.append({"suite":"requests","prompt":prompt,"act":row.act,"text":row.text,"trace":row.trace})

    answer_frames = [
        SemanticFrame.make("STATE_DEFINITION", concept="acceleration", definition="velocity change divided by time"),
        SemanticFrame.make("STATE_DEPENDENCIES", concept="speed", dependencies="distance and time"),
        SemanticFrame.make("STATE_KNOWN", concept="density"),
        SemanticFrame.make("STATE_UNKNOWN", concept="zorbin"),
        SemanticFrame.make("STATE_COMPARE", left="speed", right="density", contrast="they have different direct dependencies"),
    ]
    variant_counts: List[int] = []
    exact = total = 0
    for frame in answer_frames:
        rep = s.bidirectional_v16.roundtrip(frame, 20)
        variant_counts.append(int(rep["count"]))
        for row in rep["generations"]:
            total += 1; exact += int(row["exact"])
            if not row["exact"] and len(failures) < 30:
                failures.append({"suite":"roundtrip","frame":frame.to_dict(),"row":row})

    unknown = s.talk("what is glorpnax")
    unknown_ok = int(unknown.act == "CLARIFY" and "glorpnax" in unknown.text.lower() and "acceleration" not in unknown.text.lower())

    taught = s.talk("fluxion means acceleration")
    learned = s.talk("what is fluxion")
    learn_ok = int(taught.learned and learned.act == "ANSWER_DEFINITION" and "velocity change" in learned.text.lower())

    firewall_frame = SemanticFrame.make("STATE_DEFINITION", concept="milo flux", definition="zorbin divided by quux")
    generated = s.bidirectional_v16.generate(firewall_frame, 20)
    forbidden = {"acceleration", "speed", "density", "force", "pressure", "momentum"}
    firewall_ok = int(bool(generated) and all(not any(term in text.lower() for term in forbidden) for text in generated))

    # Frozen before its first execution. This split recombines lexical cues that
    # occur in TRAIN but puts them in new syntax. It is not expanded after seeing
    # failures; once this report is inspected it becomes development evidence.
    recombine_total = recombine_ok = 0
    teacher = s.bidirectional_teacher_v16
    for op, templates in RECOMBINATION_SPLIT.items():
        frame = teacher.frame_for(op)
        values = frame.slot_dict()
        for template in templates:
            text = template.format(**values)
            parsed, _, _ = s.language_v16.parse(text)
            recombine_total += 1; recombine_ok += int(parsed == frame)
            if parsed != frame and len(failures) < 30:
                failures.append({"suite":"recombination","text":text,"expected":frame.to_dict(),"parsed":None if parsed is None else parsed.to_dict()})

    # Original DEV intentionally contains genuinely unseen lexical material. It
    # remains diagnostic; zero-shot meaning for unknown words is not assumed.
    dev_total = dev_ok = 0
    for op, templates in teacher.DEV.items():
        frame = teacher.frame_for(op)
        values = frame.slot_dict()
        for template in templates:
            text = template.format(**values)
            parsed, _, _ = s.bidirectional_v16.parse(text, allowed_ops={op})
            dev_total += 1; dev_ok += int(parsed == frame)

    visual_after = s.visual.learner.episode_count
    return BidirectionalBenchmarkReport(
        request_accuracy=req_ok/max(1,len(requests)),
        generation_min_variants=min(variant_counts) if variant_counts else 0,
        generation_mean_variants=sum(variant_counts)/max(1,len(variant_counts)),
        roundtrip_exact=exact/max(1,total),
        unknown_honesty=float(unknown_ok),
        explicit_learning_transfer=float(learn_ok),
        content_firewall=float(firewall_ok),
        visual_experiences_changed=visual_after-visual_before,
        recombination_accuracy=recombine_ok/max(1,recombine_total),
        dev_parse_accuracy=dev_ok/max(1,dev_total),
        benchmark_role="development_and_architecture_contract_not_blind_final",
        failures=failures,
    )
