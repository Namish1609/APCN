from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import math

import numpy as np

from .identity import InstanceMemory, IdentityMatch


BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


def bbox_center(box: BBox) -> Point:
    x, y, w, h = box
    return float(x + .5*w), float(y + .5*h)


def point_in_box(p: Point, box: BBox) -> bool:
    x, y = p; bx, by, bw, bh = box
    return bx <= x <= bx + bw and by <= y <= by + bh


@dataclass
class Detection:
    descriptor: np.ndarray
    bbox: BBox
    category: Tuple[str, ...] = ()
    confidence: float = 1.0

    @property
    def center(self) -> Point:
        return bbox_center(self.bbox)


@dataclass
class WorldEvent:
    frame_index: int
    timestamp: float
    kind: str
    instance_id: str
    confidence: float
    detail: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "instance_id": self.instance_id,
            "confidence": self.confidence,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "WorldEvent":
        return cls(int(data["frame_index"]), float(data["timestamp"]), str(data["kind"]),
                   str(data["instance_id"]), float(data.get("confidence", 1.0)),
                   str(data.get("detail", "")))


@dataclass
class TrackBelief:
    instance_id: str
    state: str = "VISIBLE"
    center: Point = (.5, .5)
    velocity: Point = (0.0, 0.0)
    bbox: BBox = (.45, .45, .1, .1)
    last_seen_time: float = 0.0
    last_seen_frame: int = 0
    missing_frames: int = 0
    confidence: float = 1.0
    trail: List[Point] = field(default_factory=list)

    def predicted_center(self) -> Point:
        return (float(np.clip(self.center[0] + self.velocity[0], 0.0, 1.0)),
                float(np.clip(self.center[1] + self.velocity[1], 0.0, 1.0)))

    def update_visible(self, center: Point, bbox: BBox, timestamp: float, frame_index: int,
                       confidence: float) -> None:
        dx = center[0] - self.center[0]
        dy = center[1] - self.center[1]
        self.velocity = (.58*self.velocity[0] + .42*dx,
                         .58*self.velocity[1] + .42*dy)
        self.center = center
        self.bbox = bbox
        self.last_seen_time = float(timestamp)
        self.last_seen_frame = int(frame_index)
        self.missing_frames = 0
        self.state = "VISIBLE"
        self.confidence = float(np.clip(.68*self.confidence + .32*confidence, 0.0, 1.0))
        self.trail.append(center)
        if len(self.trail) > 32:
            del self.trail[:-32]

    def to_dict(self) -> Dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "state": self.state,
            "center": list(self.center),
            "velocity": list(self.velocity),
            "bbox": list(self.bbox),
            "last_seen_time": self.last_seen_time,
            "last_seen_frame": self.last_seen_frame,
            "missing_frames": self.missing_frames,
            "confidence": self.confidence,
            "trail": [list(x) for x in self.trail],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "TrackBelief":
        return cls(
            instance_id=str(data["instance_id"]),
            state=str(data.get("state", "VISIBLE")),
            center=tuple(float(x) for x in data.get("center", [.5, .5])),
            velocity=tuple(float(x) for x in data.get("velocity", [0.0, 0.0])),
            bbox=tuple(float(x) for x in data.get("bbox", [.45, .45, .1, .1])),
            last_seen_time=float(data.get("last_seen_time", 0.0)),
            last_seen_frame=int(data.get("last_seen_frame", 0)),
            missing_frames=int(data.get("missing_frames", 0)),
            confidence=float(data.get("confidence", 1.0)),
            trail=[tuple(float(v) for v in p) for p in data.get("trail", [])],
        )


@dataclass(frozen=True)
class FrameAssociation:
    detection_index: int
    instance_id: str
    name: Optional[str]
    identity_state: str
    identity_score: float
    created_new: bool


class PersistentWorldModel:
    """Sparse belief state over persistent object instances.

    Ground-truth object IDs never enter `process_frame`. Identity is inferred
    from appearance and temporal continuity. Missing detections decay into an
    explicit belief state rather than deleting the object immediately.
    """

    def __init__(self, instances: Optional[InstanceMemory] = None, *,
                 lost_after: int = 10, out_of_view_after: int = 4):
        self.instances = instances or InstanceMemory()
        self.tracks: Dict[str, TrackBelief] = {}
        self.events: List[WorldEvent] = []
        self.frame_index = -1
        self.lost_after = int(lost_after)
        self.out_of_view_after = int(out_of_view_after)

    def _emit(self, timestamp: float, kind: str, iid: str, confidence: float, detail: str = "") -> None:
        self.events.append(WorldEvent(self.frame_index, float(timestamp), kind, iid,
                                      float(np.clip(confidence, 0.0, 1.0)), detail))
        if len(self.events) > 600:
            del self.events[:-600]

    def predicted_centers(self) -> Dict[str, Point]:
        return {iid: t.predicted_center() for iid, t in self.tracks.items()
                if t.state != "LOST"}

    def teach_instance(self, name: str, descriptor: np.ndarray, bbox: BBox, *,
                       timestamp: float, category: Sequence[str] = ()) -> str:
        inst = self.instances.teach(name, descriptor, category=category)
        center = bbox_center(bbox)
        track = self.tracks.get(inst.instance_id)
        if track is None:
            track = TrackBelief(inst.instance_id, center=center, bbox=bbox,
                                last_seen_time=timestamp, last_seen_frame=max(0, self.frame_index))
            self.tracks[inst.instance_id] = track
            self._emit(timestamp, "TAUGHT", inst.instance_id, 1.0, name)
        else:
            track.update_visible(center, bbox, timestamp, max(0, self.frame_index), 1.0)
        return inst.instance_id

    def process_frame(self, detections: Sequence[Detection], *, timestamp: float,
                      occluders: Sequence[BBox] = (), auto_create: bool = True) -> List[FrameAssociation]:
        self.frame_index += 1
        predicted = self.predicted_centers()
        assigned: set[str] = set()
        results: List[FrameAssociation] = []

        # Highest-confidence detections first so one track cannot be claimed by
        # two objects in the same frame.
        order = sorted(range(len(detections)), key=lambda i: detections[i].confidence, reverse=True)
        for i in order:
            det = detections[i]
            match = self.instances.match(det.descriptor, category=det.category,
                                         center=det.center, predicted_centers=predicted)
            iid = match.instance_id
            created = False
            if iid in assigned:
                iid = None
            if iid is None and auto_create:
                inst = self.instances.create(category=det.category)
                inst.positive.observe(det.descriptor)
                iid = inst.instance_id
                match = IdentityMatch(iid, inst.name, .36, .0, "NOVEL", .36, .5)
                created = True
            if iid is None:
                continue
            assigned.add(iid)
            inst = self.instances.instances[iid]
            # Only self-reinforce reasonably confident re-identifications. This
            # avoids a low-confidence wrong match rapidly corrupting appearance.
            if not created and match.state in {"KNOWN", "PROBABLE"}:
                self.instances.reinforce(iid, det.descriptor)
            center = det.center
            if iid not in self.tracks:
                self.tracks[iid] = TrackBelief(iid, center=center, bbox=det.bbox,
                                               last_seen_time=timestamp, last_seen_frame=self.frame_index,
                                               confidence=max(.45, match.score))
                self._emit(timestamp, "APPEARED", iid, max(.45, match.score))
            else:
                track = self.tracks[iid]
                prev_state = track.state
                old = track.center
                track.update_visible(center, det.bbox, timestamp, self.frame_index,
                                     max(det.confidence, match.score))
                if prev_state in {"OCCLUDED", "OUT_OF_VIEW", "LOST"}:
                    self._emit(timestamp, "REAPPEARED", iid, match.score, prev_state)
                if math.hypot(center[0]-old[0], center[1]-old[1]) > .035:
                    self._emit(timestamp, "MOVED", iid, track.confidence)
            results.append(FrameAssociation(i, iid, inst.name, match.state,
                                            match.score, created))

        for iid, track in self.tracks.items():
            if iid in assigned:
                continue
            track.missing_frames += 1
            pred = track.predicted_center()
            old_state = track.state
            if track.missing_frames >= self.lost_after:
                track.state = "LOST"
                track.confidence *= .72
            elif any(point_in_box(pred, box) for box in occluders):
                track.state = "OCCLUDED"
                track.center = pred
                track.confidence *= .95
            elif track.missing_frames >= self.out_of_view_after or pred[0] < .03 or pred[0] > .97 or pred[1] < .03 or pred[1] > .97:
                track.state = "OUT_OF_VIEW"
                track.center = pred
                track.confidence *= .88
            else:
                track.state = "OCCLUDED"
                track.center = pred
                track.confidence *= .92
            if track.state != old_state:
                self._emit(timestamp, track.state, iid, track.confidence)
        return sorted(results, key=lambda x: x.detection_index)

    def correct_identity(self, *, wrong_instance_id: Optional[str], correct_name: str,
                         descriptor: np.ndarray, bbox: Optional[BBox] = None,
                         timestamp: float = 0.0, category: Sequence[str] = ()) -> str:
        correct = self.instances.correct(wrong_instance_id=wrong_instance_id,
                                         correct_name=correct_name,
                                         descriptor=descriptor, category=category)
        if bbox is not None:
            center = bbox_center(bbox)
            track = self.tracks.get(correct.instance_id)
            if track is None:
                self.tracks[correct.instance_id] = TrackBelief(
                    correct.instance_id, center=center, bbox=bbox,
                    last_seen_time=timestamp, last_seen_frame=max(0, self.frame_index))
            else:
                track.update_visible(center, bbox, timestamp, max(0, self.frame_index), 1.0)
        self._emit(timestamp, "CORRECTED_IDENTITY", correct.instance_id, 1.0, correct_name)
        return correct.instance_id

    @staticmethod
    def _zone(center: Point) -> str:
        x, y = center
        horiz = "left" if x < .34 else "right" if x > .66 else "center"
        vert = "top" if y < .34 else "bottom" if y > .66 else "middle"
        return f"{vert}-{horiz}"

    def where(self, name_or_id: str) -> Dict[str, object]:
        inst = self.instances.by_name(name_or_id)
        iid = inst.instance_id if inst is not None else name_or_id
        track = self.tracks.get(iid)
        if track is None:
            return {"known": False, "answer": f"I do not have a tracked belief for {name_or_id}."}
        name = self.instances.instances.get(iid).name if iid in self.instances.instances else None
        label = name or iid
        if track.state == "VISIBLE":
            answer = f"{label} is currently visible in the {self._zone(track.center)} region."
        elif track.state == "OCCLUDED":
            answer = (f"{label} is not currently visible. I believe it is occluded near the "
                      f"{self._zone(track.center)} region based on its last trajectory.")
        elif track.state == "OUT_OF_VIEW":
            answer = (f"{label} is currently out of view. Its last tracked location was near the "
                      f"{self._zone(track.center)} region.")
        else:
            answer = (f"I have lost the current location of {label}. The last tracked region was "
                      f"{self._zone(track.center)}.")
        return {
            "known": True,
            "instance_id": iid,
            "name": name,
            "state": track.state,
            "center": list(track.center),
            "velocity": list(track.velocity),
            "confidence": float(track.confidence),
            "last_seen_frame": track.last_seen_frame,
            "last_seen_time": track.last_seen_time,
            "answer": answer,
        }

    def memory_summary(self) -> Dict[str, object]:
        states: Dict[str, int] = {}
        for t in self.tracks.values():
            states[t.state] = states.get(t.state, 0) + 1
        return {
            "instance_memory": self.instances.memory_summary(),
            "tracks": len(self.tracks),
            "states": states,
            "events_retained": len(self.events),
            "max_events_retained": 600,
            "raw_video_frames_retained": 0,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "instances": self.instances.to_dict(),
            "tracks": {k: v.to_dict() for k, v in self.tracks.items()},
            "events": [e.to_dict() for e in self.events],
            "frame_index": self.frame_index,
            "lost_after": self.lost_after,
            "out_of_view_after": self.out_of_view_after,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PersistentWorldModel":
        obj = cls(
            instances=InstanceMemory.from_dict(data.get("instances", {})),
            lost_after=int(data.get("lost_after", 10)),
            out_of_view_after=int(data.get("out_of_view_after", 4)),
        )
        obj.tracks = {k: TrackBelief.from_dict(v) for k, v in data.get("tracks", {}).items()}
        obj.events = [WorldEvent.from_dict(x) for x in data.get("events", [])][-600:]
        obj.frame_index = int(data.get("frame_index", -1))
        return obj

    def save(self, path: str | Path) -> None:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PersistentWorldModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
