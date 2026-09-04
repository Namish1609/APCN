from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import json
import random

import numpy as np

from .session import CognitiveSessionV13
from .temporal_teacher import SyntheticInstanceSpec, TemporalSceneTeacher
from .world import PersistentWorldModel


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
        return {"identity_accuracy": self.identity_accuracy, "similar_instance_accuracy": self.similar_instance_accuracy,
                "unknown_rejection": self.unknown_rejection, "reappearance_recovery": self.reappearance_recovery,
                "occlusion_state_accuracy": self.occlusion_state_accuracy, "correction_success": self.correction_success,
                "bounded_memory_ok": self.bounded_memory_ok, "details": self.details}


def _teach_views(session, teacher, spec, *, views, rng):
    iid = ""
    for i in range(views):
        frame = teacher.render(spec, center=(.22+.56*rng.random(),.24+.50*rng.random()), scale=rng.uniform(.27,.38),
                               angle=rng.uniform(-34,34), brightness=rng.uniform(.72,1.18),
                               background_seed=9000+i+rng.randrange(100000))
        iid = str(session.teach_named_instance(spec.name,frame.image,bbox=frame.bbox,category=spec.category,
                  timestamp=float(i),attention_mask=frame.attention_mask)["instance_id"])
    return iid


def _descriptor(session, frame):
    return session.descriptor(frame.image,bbox=frame.bbox,attention_mask=frame.attention_mask)


def run_v13_benchmark(seed=13013, *, visual_bootstrap=540, teach_views=10, test_views=60):
    rng=random.Random(seed); teacher=TemporalSceneTeacher(seed=seed); session=CognitiveSessionV13(seed)
    session.visual.learner.representation_learning_until=max(visual_bootstrap,720); session._balanced_visual_bootstrap(visual_bootstrap)
    a,b=teacher.similar_pair(); a_id=_teach_views(session,teacher,a,views=teach_views,rng=rng); b_id=_teach_views(session,teacher,b,views=teach_views,rng=rng)

    total=correct=0; per_name={a.name:[0,0],b.name:[0,0]}; familiar_states={}; failures={"rejected":0,"wrong_twin":0}
    score_sum=margin_sum=0.0
    for spec,iid in ((a,a_id),(b,b_id)):
        for j in range(test_views):
            frame=teacher.render(spec,center=(.14+.72*rng.random(),.18+.62*rng.random()),scale=rng.uniform(.25,.40),
                                 angle=rng.uniform(-43,43),brightness=rng.uniform(.62,1.28),background_seed=seed+20000+j+total)
            m=session.world.instances.match(_descriptor(session,frame),category=spec.category)
            total+=1; per_name[spec.name][1]+=1; score_sum+=m.score; margin_sum+=m.margin; familiar_states[m.state]=familiar_states.get(m.state,0)+1
            if m.instance_id==iid: correct+=1; per_name[spec.name][0]+=1
            elif m.instance_id is None: failures["rejected"]+=1
            else: failures["wrong_twin"]+=1
    identity=correct/max(1,total); similar=min(per_name[a.name][0]/per_name[a.name][1],per_name[b.name][0]/per_name[b.name][1])

    novel=SyntheticInstanceSpec("novel_bottle","blue","rectangle",13991,("bottle",)); novel_ok=0; novel_states={}; ns=nm=0.0; nn=0
    for j in range(max(20,test_views//2)):
        frame=teacher.render(novel,center=(.18+.64*rng.random(),.2+.58*rng.random()),scale=rng.uniform(.26,.39),
                             angle=rng.uniform(-40,40),brightness=rng.uniform(.68,1.22),background_seed=seed+50000+j)
        m=session.world.instances.match(_descriptor(session,frame),category=novel.category)
        novel_states[m.state]=novel_states.get(m.state,0)+1; ns+=m.score; nm+=m.margin; nn+=1
        if m.state in {"NOVEL","AMBIGUOUS"}: novel_ok+=1
    unknown=novel_ok/max(1,nn)

    session.world=PersistentWorldModel(session.world.instances,lost_after=9,out_of_view_after=5); temporal=[]
    for k,xp in enumerate([.20,.30,.40,.50]):
        frame=teacher.render(a,center=(xp,.52),scale=.33,angle=8+k*2,brightness=.92,background_seed=seed+70000+k)
        row=session.observe_object(frame.image,bbox=frame.bbox,category=a.category,timestamp=100+k,
                                   attention_mask=frame.attention_mask,auto_create=False); temporal.append(row.get("instance_id"))
    occ=(.46,.28,.34,.48); occ_ok=0
    for k in range(3):
        session.observe_absence(timestamp=104+k,occluders=[occ]); t=session.world.tracks.get(a_id)
        if t is not None and t.state=="OCCLUDED": occ_ok+=1
    reframe=teacher.render(a,center=(.70,.52),scale=.32,angle=15,brightness=1.04,background_seed=seed+71000)
    re=session.observe_object(reframe.image,bbox=reframe.bbox,category=a.category,timestamp=108,attention_mask=reframe.attention_mask,auto_create=False)
    reappear=1.0 if (re.get("instance_id")==a_id and a_id in session.world.tracks and session.world.tracks[a_id].state=="VISIBLE" and
                     any(e.kind=="REAPPEARED" and e.instance_id==a_id for e in session.world.events)) else 0.0

    cf=teacher.render(b,center=(.54,.42),scale=.35,angle=-17,brightness=.88,background_seed=seed+72000); xc=_descriptor(session,cf)
    ba=session.world.instances.instances[a_id].appearance_score(xc); bb=session.world.instances.instances[b_id].appearance_score(xc)
    session.world.correct_identity(wrong_instance_id=a_id,correct_name=b.name,descriptor=xc,bbox=cf.bbox,timestamp=120,category=b.category)
    aa=session.world.instances.instances[a_id].appearance_score(xc); ab=session.world.instances.instances[b_id].appearance_score(xc)
    correction=1.0 if ab>aa and aa<ba else 0.0
    mem=session.world.memory_summary(); im=mem["instance_memory"]
    bounded=(im["raw_frames_retained"]==0 and im["raw_descriptors_retained"]==0 and mem["raw_video_frames_retained"]==0 and
             all(len(x.positive.prototypes)<=session.world.instances.max_views for x in session.world.instances.instances.values()))
    return V13BenchmarkResult(identity,similar,unknown,reappear,occ_ok/3.0,correction,bool(bounded),{
        "per_instance":per_name,"familiar_states":familiar_states,"failure_types":failures,
        "mean_familiar_score":score_sum/max(1,total),"mean_familiar_margin":margin_sum/max(1,total),
        "novel_states":novel_states,"mean_novel_score":ns/max(1,nn),"mean_novel_margin":nm/max(1,nn),
        "temporal_matches":temporal,"world_memory":mem,"where_after_reappearance":session.where(a.name),
        "correction_scores":{"before_wrong":ba,"before_correct":bb,"after_wrong":aa,"after_correct":ab}})


def main():
    print(json.dumps(run_v13_benchmark().to_dict(),indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
