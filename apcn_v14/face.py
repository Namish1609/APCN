from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import json

import cv2
import numpy as np

from apcn_v13.appearance import FineAppearanceEncoder
from apcn_v13.identity import IdentityMatch, InstanceMemory

BBox = Tuple[float, float, float, float]


class ClassicalFaceLocator:
    """Optional local face locator; not a face-identity encoder."""
    def __init__(self):
        path=Path(cv2.data.haarcascades)/"haarcascade_frontalface_default.xml"; self.path=str(path)
        self.cascade=cv2.CascadeClassifier(self.path) if path.exists() else None
        if self.cascade is not None and self.cascade.empty(): self.cascade=None
    @property
    def available(self)->bool: return self.cascade is not None
    def locate_largest(self,frame: np.ndarray)->Optional[BBox]:
        if self.cascade is None or frame is None or frame.ndim!=3: return None
        gray=cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        faces=self.cascade.detectMultiScale(gray,scaleFactor=1.10,minNeighbors=5,minSize=(56,56))
        if len(faces)==0: return None
        x,y,w,h=max(faces,key=lambda r:int(r[2])*int(r[3])); H,W=frame.shape[:2]
        return float(x/W),float(y/H),float(w/W),float(h/H)
    def summary(self)->Dict[str,object]:
        return {"available":self.available,"type":"OpenCV Haar cascade face locator","neural":False,"downloads_required":False,"identity_role":False}


class SelfFaceMemory:
    """Opt-in single-person local face memory for desktop self-testing.

    No neural face embedding model is used. One enrolled identity is represented
    by bounded aggregate APCN appearance prototypes. Raw frames/crops are never
    serialized. Optional negative examples improve unknown-person rejection.

    V0.14 uses a FACE-SPECIFIC deterministic normalization before the generic
    V0.13 fine-appearance extractor: a tighter square crop, grayscale structure,
    and neutralized corner/background pixels. This avoids asking object color or
    room background to carry personal identity. No learned weights are involved.
    """
    VERSION="APCN-V0.14-LOCAL-SELF-FACE-MEMORY"
    def __init__(self,max_views: int=12):
        # Allocate most descriptor capacity to luminance/gradient structure.
        # Chromaticity remains present only as a tiny constant compatibility block
        # because the canonical face crop is grayscale.
        self.encoder=FineAppearanceEncoder(
            normalized_size=56,luminance_size=24,chromatic_size=4,
            gradient_size=20,occupancy_size=6)
        self.memory=InstanceMemory(max_views=max_views,strong_threshold=.58,probable_threshold=.46,ambiguity_margin=.045)
        self.locator=ClassicalFaceLocator(); self.enrolled_name: Optional[str]=None; self.enroll_observations=0; self.negative_observations=0; self.recognition_queries=0
    @staticmethod
    def _clip_bbox(bbox:BBox)->BBox:
        x,y,w,h=[float(v) for v in bbox]; x=min(max(x,0.0),.999); y=min(max(y,0.0),.999); w=min(max(w,.01),1.0-x); h=min(max(h,.01),1.0-y); return x,y,w,h
    @classmethod
    def _square_crop(cls,frame: np.ndarray,bbox:BBox,pad: float=.045)->np.ndarray:
        if frame is None or frame.ndim!=3 or frame.shape[2]!=3: raise ValueError("expected HxWx3 BGR frame")
        H,W=frame.shape[:2]; x,y,bw,bh=cls._clip_bbox(bbox); cx=(x+bw/2)*W; cy=(y+bh/2)*H; side=max(bw*W,bh*H)*(1+2*pad)
        x0=int(np.floor(cx-side/2)); y0=int(np.floor(cy-side/2)); x1=int(np.ceil(cx+side/2)); y1=int(np.ceil(cy+side/2)); sx0,sy0=max(0,x0),max(0,y0); sx1,sy1=min(W,x1),min(H,y1)
        crop=frame[sy0:sy1,sx0:sx1]
        if crop.size==0: raise ValueError("face crop is empty")
        top,left=sy0-y0,sx0-x0; bottom,right=y1-sy1,x1-sx1
        if top or bottom or left or right: crop=cv2.copyMakeBorder(crop,top,bottom,left,right,cv2.BORDER_REPLICATE)
        side_px=max(crop.shape[:2])
        if crop.shape[0]!=crop.shape[1]:
            dh=side_px-crop.shape[0]; dw=side_px-crop.shape[1]; crop=cv2.copyMakeBorder(crop,dh//2,dh-dh//2,dw//2,dw-dw//2,cv2.BORDER_REPLICATE)
        return np.ascontiguousarray(crop)
    @staticmethod
    def _canonical_structure(crop: np.ndarray)->np.ndarray:
        """Suppress illumination/color/background nuisance without face landmarks."""
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        h,w=gray.shape
        # A circular central support is deliberately symmetric so it does not
        # create an artificial principal orientation in FineAppearanceEncoder.
        mask=np.zeros((h,w),dtype=np.uint8)
        radius=max(2,int(round(.47*min(h,w))))
        cv2.circle(mask,(w//2,h//2),radius,255,-1,cv2.LINE_AA)
        inside=gray[mask>0]
        fill=int(np.median(inside)) if inside.size else int(np.median(gray))
        structural=gray.copy(); structural[mask==0]=fill
        # Gentle local normalization; unlike face embeddings this has no learned
        # parameters and does not encode identity by itself.
        structural=cv2.GaussianBlur(structural,(3,3),0)
        return cv2.cvtColor(structural,cv2.COLOR_GRAY2BGR)
    def descriptor(self,frame: np.ndarray,bbox:BBox)->np.ndarray:
        crop=self._canonical_structure(self._square_crop(frame,bbox)); mask=np.ones(crop.shape[:2],dtype=np.uint8)*255
        return self.encoder.extract(crop,attention_mask=mask,bbox=(0.0,0.0,1.0,1.0))
    def auto_bbox(self,frame: np.ndarray)->Optional[BBox]: return self.locator.locate_largest(frame)
    def enroll(self,name: str,frame: np.ndarray,bbox:BBox)->Dict[str,object]:
        name=str(name).strip() or "me"
        if self.enrolled_name is not None and name.lower()!=self.enrolled_name.lower(): raise ValueError(f"SelfFaceMemory is single-identity; already enrolled as {self.enrolled_name!r}")
        inst=self.memory.teach(name,self.descriptor(frame,bbox),category=("self_face",)); self.enrolled_name=name; self.enroll_observations+=1
        return {"name":name,"instance_id":inst.instance_id,"views":inst.positive.observations,"prototype_modes":len(inst.positive.prototypes)}
    def recognize(self,frame: np.ndarray,bbox:BBox)->Dict[str,object]:
        self.recognition_queries+=1
        if self.enrolled_name is None: return {"enrolled":False,"match":False,"state":"UNENROLLED","score":0.0}
        result:IdentityMatch=self.memory.match(self.descriptor(frame,bbox),category=("self_face",)); committed=result.instance_id is not None and result.name is not None
        return {"enrolled":True,"match":bool(committed),"name":result.name if committed else None,"state":result.state,"score":result.score,"margin":result.margin,"appearance_score":result.appearance_score}
    def mark_not_me(self,frame: np.ndarray,bbox:BBox)->Dict[str,object]:
        if self.enrolled_name is None: raise ValueError("enroll your face before adding a negative example")
        inst=self.memory.by_name(self.enrolled_name); assert inst is not None; x=self.descriptor(frame,bbox); before=inst.appearance_score(x); inst.negative.observe(x); inst.correction_events+=1; self.negative_observations+=1; after=inst.appearance_score(x)
        return {"name":self.enrolled_name,"score_before":before,"score_after":after,"negative_examples":self.negative_observations}
    def summary(self)->Dict[str,object]:
        return {"version":self.VERSION,"single_identity":True,"enrolled_name":self.enrolled_name,"enroll_observations":self.enroll_observations,"negative_observations":self.negative_observations,"recognition_queries":self.recognition_queries,"instance_memory":self.memory.memory_summary(),"locator":self.locator.summary(),"neural_face_encoder":False,"deterministic_structural_normalization":True,"raw_face_images_retained":0,"raw_camera_frames_retained":0}
    def to_dict(self)->Dict[str,object]:
        return {"version":self.VERSION,"enrolled_name":self.enrolled_name,"enroll_observations":self.enroll_observations,"negative_observations":self.negative_observations,"recognition_queries":self.recognition_queries,"memory":self.memory.to_dict(),"raw_face_images_retained":0,"raw_camera_frames_retained":0}
    @classmethod
    def from_dict(cls,data: Dict[str,object])->"SelfFaceMemory":
        obj=cls(int(data.get("memory",{}).get("max_views",12))); obj.enrolled_name=data.get("enrolled_name"); obj.enroll_observations=int(data.get("enroll_observations",0)); obj.negative_observations=int(data.get("negative_observations",0)); obj.recognition_queries=int(data.get("recognition_queries",0)); obj.memory=InstanceMemory.from_dict(data.get("memory",{})); return obj
    def save(self,path: str|Path)->None:
        p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(self.to_dict(),indent=2),encoding="utf-8")
    @classmethod
    def load(cls,path: str|Path)->"SelfFaceMemory": return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
