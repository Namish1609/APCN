from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import random

import numpy as np

from .session import CognitiveSessionV13
from .temporal_teacher import SyntheticInstanceSpec, TemporalSceneTeacher


@dataclass
class V13BenchmarkResult:
    identity_accuracy: float
    similar_instance_accuracy: float
    unknown_rejection: float
    reappearance_recovery: float
    occlusion_state_accuracy: float
    correction_success: float
    bounded_memory_ok: bool
    details: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "identity_accuracy": self.identity_accuracy,
            "similar_instance_accuracy": self.similar_instance_accuracy,
            "unknown_rejection": self.unknown_rejection,
            "reappearance_recovery": self.reappearance_recovery,
            "occlusion_state_accuracy": self.occlusion_state_accuracy,
            "correction_success": self.correction_success,
            "bounded_memory_ok": self.bounded_memory_ok,
            "details": self.details,
        }


def _teach_views(session: CognitiveSessionV13, teacher: TemporalSceneTeacher,
                 spec: SyntheticInstanceSpec, *, views: int, rng: random.Random) -> str:
    iid = ""
    for i in range(views):
        center = (.22 + .56*rng.random(), .24 + .50*rng.random())
        angle = rng.uniform(-34,34)
        scale = rng.uniform(.27,.38)
        brightness = rng.uniform(.72,1.18)
        frame = teacher.render(spec, center=center, scale=scale, angle=angle,
                               brightness=brightness, background_seed=9000+i+rng.randrange(100000))
        row = session.teach_named_instance(spec.name, frame.image, bbox=frame.bbox,
                                           category=spec.category, timestamp=float(i),
                                           attention_mask=frame.attention_mask)
        iid = str(row["instance_id"])
    return iid


def _descriptor(session: CognitiveSessionV13, frame) -> np.ndarray:
    return session.descriptor(frame.image, bbox=frame.bbox, attention_mask=frame.attention_mask)


def run_v13_benchmark(seed: int = 13013, *, visual_bootstrap: int = 540,
                      teach_views: int = 10, test_views: int = 60) -> V13BenchmarkResult:
    rng = random.Random(seed)
    teacher = TemporalSceneTeacher(seed=seed)
    session = CognitiveSessionV13(seed)
    session.visual.learner.representation_learning_until = max(visual_bootstrap, 720)
    session._balanced_visual_bootstrap(visual_bootstrap)

    a, b = teacher.similar_pair()
    a_id = _teach_views(session, teacher, a, views=teach_views, rng=rng)
    b_id = _teach_views(session, teacher, b, views=teach_views, rng=rng)

    total = 0; correct = 0
    per_name = {a.name: [0,0], b.name:[0,0]}
    for spec, iid in ((a,a_id),(b,b_id)):
        for j in range(test_views):
            frame = teacher.render(
                spec,
                center=(.14+.72*rng.random(), .18+.62*rng.random()),
                scale=rng.uniform(.25,.40), angle=rng.uniform(-43,43),
                brightness=rng.uniform(.62,1.28), background_seed=seed+20000+j+total,
            )
            x = _descriptor(session, frame)
            match = session.world.instances.match(x, category=spec.category)
            total += 1; per_name[spec.name][1] += 1
            if match.instance_id == iid:
                correct += 1; per_name[spec.name][0] += 1
    identity_accuracy = correct / max(1,total)
    similar_accuracy = min(per_name[a.name][0]/max(1,per_name[a.name][1]),
                           per_name[b.name][0]/max(1,per_name[b.name][1]))

    # A third visually distinct instance is never taught. Correct behavior is
    # to avoid a confident KNOWN assignment to either familiar bottle.
    novel = SyntheticInstanceSpec("novel_bottle", "blue", "rectangle", 13991, ("bottle",))
    novel_ok = 0
    novel_states: Dict[str,int] = {}
    for j in range(max(20,test_views//2)):
        frame = teacher.render(novel, center=(.18+.64*rng.random(), .2+.58*rng.random()),
                               scale=rng.uniform(.26,.39), angle=rng.uniform(-40,40),
                               brightness=rng.uniform(.68,1.22), background_seed=seed+50000+j)
        m = session.world.instances.match(_descriptor(session,frame), category=novel.category)
        novel_states[m.state] = novel_states.get(m.state,0)+1
        if m.state in {"NOVEL","AMBIGUOUS"}:
            novel_ok += 1
    unknown_rejection = novel_ok / max(1,sum(novel_states.values()))

    # Temporal identity + occlusion. No teacher ID is given during these frames.
    session.world.frame_index = 0
    x_positions = [.20,.28,.36,.43]
    for k,xp in enumerate(x_positions):
        frame = teacher.render(a, center=(xp,.52), scale=.33, angle=8+k*2,
                               brightness=.92, background_seed=seed+70000+k)
        session.observe_object(frame.image, bbox=frame.bbox, category=a.category,
                               timestamp=100+k, attention_mask=frame.attention_mask,
                               auto_create=False)
    occ = (.40,.28,.34,.48)
    occ_correct = 0
    for k in range(3):
        session.observe_absence(timestamp=104+k, occluders=[occ])
        state = session.world.tracks[a_id].state
        if state == "OCCLUDED": occ_correct += 1
    occlusion_accuracy = occ_correct/3.0

    re_frame = teacher.render(a, center=(.64,.52), scale=.32, angle=15,
                              brightness=1.04, background_seed=seed+71000)
    re = session.observe_object(re_frame.image, bbox=re_frame.bbox, category=a.category,
                                timestamp=108, attention_mask=re_frame.attention_mask,
                                auto_create=False)
    reappearance = 1.0 if re.get("instance_id") == a_id and session.world.tracks[a_id].state == "VISIBLE" else 0.0

    # Immediate human correction: penalize the wrong identity and strengthen the
    # correct one using a single current observation.
    correction_frame = teacher.render(b, center=(.54,.42), scale=.35, angle=-17,
                                      brightness=.88, background_seed=seed+72000)
    xcorr = _descriptor(session, correction_frame)
    before_a = session.world.instances.instances[a_id].appearance_score(xcorr)
    before_b = session.world.instances.instances[b_id].appearance_score(xcorr)
    session.world.correct_identity(wrong_instance_id=a_id, correct_name=b.name,
                                   descriptor=xcorr, bbox=correction_frame.bbox,
                                   timestamp=120, category=b.category)
    after_a = session.world.instances.instances[a_id].appearance_score(xcorr)
    after_b = session.world.instances.instances[b_id].appearance_score(xcorr)
    correction_success = 1.0 if after_b > after_a and after_a < before_a else 0.0

    mem = session.world.memory_summary()
    inst_mem = mem["instance_memory"]
    bounded = (
        inst_mem["raw_frames_retained"] == 0 and
        inst_mem["raw_descriptors_retained"] == 0 and
        all(len(x.positive.prototypes) <= session.world.instances.max_views
            for x in session.world.instances.instances.values()) and
        mem["raw_video_frames_retained"] == 0
    )
    return V13BenchmarkResult(
        identity_accuracy=identity_accuracy,
        similar_instance_accuracy=similar_accuracy,
        unknown_rejection=unknown_rejection,
        reappearance_recovery=reappearance,
        occlusion_state_accuracy=occlusion_accuracy,
        correction_success=correction_success,
        bounded_memory_ok=bool(bounded),
        details={
            "per_instance": per_name,
            "novel_states": novel_states,
            "world_memory": mem,
            "where_after_reappearance": session.where(a.name),
            "correction_scores": {
                "before_wrong": before_a, "before_correct": before_b,
                "after_wrong": after_a, "after_correct": after_b,
            },
        },
    )


def main() -> int:
    result = run_v13_benchmark()
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
