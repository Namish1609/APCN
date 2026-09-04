from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import json

import cv2
import numpy as np

from apcn_v10.definitions import ConceptStore, DefinitionCurriculum
from apcn_v10.query import KnowledgeQueryEngine
from apcn_v11.concept_graph import UnifiedConceptGraph
from apcn_v11.consolidation import ConsolidationEngine
from apcn_v11.discourse import DiscourseEntityRegistry
from apcn_v11.error_memory import ErrorMemory
from apcn_v12.language import AdaptiveLanguageSessionV12, SemanticLanguageLearnerV12
from apcn_v12.session import CognitiveSessionV12, TrainingSessionV12
from apcn_v12.visual import SelfOrganizingVisualLearner

from .world import BBox, Detection, PersistentWorldModel


class CognitiveSessionV13(CognitiveSessionV12):
    VERSION = "0.13.0"

    def __init__(self, seed: int = 13):
        super().__init__(seed)
        self.seed = seed
        self.world = PersistentWorldModel()
        self.world_test_history = []
        self._last_descriptor: Optional[np.ndarray] = None
        self._last_bbox: Optional[BBox] = None
        self._last_category: Tuple[str, ...] = ()
        self._last_association = None

    @staticmethod
    def mask_from_bbox(image: np.ndarray, bbox: BBox) -> np.ndarray:
        h, w = image.shape[:2]
        x, y, bw, bh = bbox
        x0 = int(np.clip(round(x*w), 0, w-1)); y0 = int(np.clip(round(y*h), 0, h-1))
        x1 = int(np.clip(round((x+bw)*w), x0+1, w)); y1 = int(np.clip(round((y+bh)*h), y0+1, h))
        mask = np.zeros((h,w), dtype=np.uint8)
        mask[y0:y1,x0:x1] = 255
        return mask

    def descriptor(self, image: np.ndarray, *, bbox: Optional[BBox] = None,
                   attention_mask: Optional[np.ndarray] = None,
                   adapt_representation: bool = False) -> np.ndarray:
        if attention_mask is None:
            if bbox is None:
                raise ValueError("bbox or attention_mask is required")
            attention_mask = self.mask_from_bbox(image, bbox)
        if adapt_representation:
            self.visual.learner.patch_sensor.learn(image, attention_mask)
        return self.visual.learner.patch_sensor.extract(image, attention_mask)

    def infer_grounded_category(self, descriptor: np.ndarray) -> Tuple[str, ...]:
        learner = self.visual.learner
        colors = list(self.visual.teacher.color_words)
        shapes = list(self.visual.teacher.shape_words)
        color = learner.best_of(colors, descriptor)[0] if learner.token_stats else None
        shape = learner.best_of(shapes, descriptor)[0] if learner.token_stats else None
        return tuple(x for x in (color, shape) if x)

    def teach_named_instance(self, name: str, image: np.ndarray, *, bbox: BBox,
                             category: Sequence[str] = (), timestamp: float = 0.0,
                             attention_mask: Optional[np.ndarray] = None,
                             adapt_representation: bool = False) -> Dict[str, object]:
        x = self.descriptor(image, bbox=bbox, attention_mask=attention_mask,
                            adapt_representation=adapt_representation)
        cat = tuple(str(v).lower() for v in category) or self.infer_grounded_category(x)
        iid = self.world.teach_instance(name, x, bbox, timestamp=timestamp, category=cat)
        self._remember_working_observation(x, bbox, cat, None)
        return {"instance_id": iid, "name": name, "category": list(cat),
                "views": self.world.instances.instances[iid].positive.observations}

    def observe_object(self, image: np.ndarray, *, bbox: BBox,
                       category: Sequence[str] = (), timestamp: float = 0.0,
                       attention_mask: Optional[np.ndarray] = None,
                       occluders: Sequence[BBox] = (), auto_create: bool = True) -> Dict[str, object]:
        x = self.descriptor(image, bbox=bbox, attention_mask=attention_mask, adapt_representation=False)
        cat = tuple(str(v).lower() for v in category) or self.infer_grounded_category(x)
        rows = self.world.process_frame([Detection(x, bbox, cat)], timestamp=timestamp,
                                        occluders=occluders, auto_create=auto_create)
        assoc = rows[0] if rows else None
        self._remember_working_observation(x, bbox, cat, assoc)
        if assoc is None:
            return {"matched": False, "state": "NOVEL"}
        return {"matched": True, "instance_id": assoc.instance_id, "name": assoc.name,
                "identity_state": assoc.identity_state, "score": assoc.identity_score,
                "created_new": assoc.created_new, "category": list(cat)}

    def observe_absence(self, *, timestamp: float, occluders: Sequence[BBox] = ()) -> None:
        self.world.process_frame([], timestamp=timestamp, occluders=occluders, auto_create=False)

    def correct_last_identity(self, correct_name: str, *, wrong_instance_id: Optional[str] = None,
                              timestamp: float = 0.0) -> Dict[str, object]:
        if self._last_descriptor is None or self._last_bbox is None:
            raise ValueError("no current observation is available to correct")
        if wrong_instance_id is None and self._last_association is not None:
            wrong_instance_id = self._last_association.instance_id
        iid = self.world.correct_identity(wrong_instance_id=wrong_instance_id,
                                          correct_name=correct_name,
                                          descriptor=self._last_descriptor,
                                          bbox=self._last_bbox,
                                          timestamp=timestamp,
                                          category=self._last_category)
        return {"corrected": True, "instance_id": iid, "name": correct_name}

    def _remember_working_observation(self, descriptor: np.ndarray, bbox: BBox,
                                      category: Sequence[str], association) -> None:
        # Working state only. It is intentionally excluded from checkpoints.
        self._last_descriptor = np.asarray(descriptor, dtype=np.float64).copy()
        self._last_bbox = tuple(float(x) for x in bbox)
        self._last_category = tuple(str(x) for x in category)
        self._last_association = association

    def where(self, name_or_id: str) -> Dict[str, object]:
        return self.world.where(name_or_id)

    def memory_audit(self) -> Dict[str, object]:
        base = super().memory_audit()
        base["v013_world_memory"] = self.world.memory_summary()
        base["v013_working_descriptor_persisted"] = False
        return base

    @classmethod
    def from_v12_checkpoint(cls, output_dir: str | Path = "outputs/v0_12", *, seed: int = 13) -> "CognitiveSessionV13":
        old = CognitiveSessionV12.load_checkpoint(output_dir, seed=seed)
        obj = cls(seed)
        obj.visual = old.visual
        obj.language = old.language
        obj.concepts = old.concepts
        obj.definitions = old.definitions
        obj.query = old.query
        obj.graph = old.graph
        obj.errors = old.errors
        obj.consolidation = ConsolidationEngine(obj.errors)
        obj.visual_test_history = list(old.visual_test_history)
        obj.language_test_history = list(old.language_test_history)
        obj.test_history = obj.language_test_history
        obj.consolidation_history = list(old.consolidation_history)
        obj.v012_bootstrap_experiences = old.v012_bootstrap_experiences
        return obj

    @classmethod
    def load_checkpoint(cls, output_dir: str | Path = "outputs/v0_13", *, seed: int = 13) -> "CognitiveSessionV13":
        out = Path(output_dir)
        obj = cls(seed)
        vp = out / "visual_memory_v0_13.json"
        if vp.exists(): obj.visual = TrainingSessionV12(seed, learner=SelfOrganizingVisualLearner.load(vp))
        lp = out / "language_memory_v0_13.json"
        if lp.exists(): obj.language = AdaptiveLanguageSessionV12(seed, learner=SemanticLanguageLearnerV12.load(lp))
        cp = out / "concept_store_v0_13.json"
        if cp.exists():
            obj.concepts = ConceptStore.load(cp); obj.definitions = DefinitionCurriculum(obj.concepts); obj.query = KnowledgeQueryEngine(obj.concepts)
        ep = out / "error_memory_v0_13.json"
        if ep.exists(): obj.errors = ErrorMemory.load(ep); obj.consolidation = ConsolidationEngine(obj.errors)
        dp = out / "discourse_state_v0_13.json"
        if dp.exists(): obj.language.discourse = DiscourseEntityRegistry.load(dp)
        gp = out / "unified_concept_graph_v0_13.json"
        if gp.exists():
            try: obj.graph = UnifiedConceptGraph.load(gp)
            except Exception: obj.graph = UnifiedConceptGraph()
        wp = out / "world_memory_v0_13.json"
        if wp.exists(): obj.world = PersistentWorldModel.load(wp)
        sp = out / "session_v0_13.json"
        if sp.exists():
            state = json.loads(sp.read_text(encoding="utf-8"))
            obj.visual_test_history = list(state.get("visual_test_history", []))
            obj.language_test_history = list(state.get("language_test_history", []))
            obj.test_history = obj.language_test_history
            obj.consolidation_history = list(state.get("consolidation_history", []))
            obj.world_test_history = list(state.get("world_test_history", []))
            obj.v012_bootstrap_experiences = int(state.get("v012_bootstrap_experiences", 0))
        obj.sync_graph()
        return obj

    def save(self, output_dir: str | Path = "outputs/v0_13") -> Dict[str, str]:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        paths = {
            "visual": out / "visual_memory_v0_13.json",
            "language": out / "language_memory_v0_13.json",
            "concepts": out / "concept_store_v0_13.json",
            "graph": out / "unified_concept_graph_v0_13.json",
            "errors": out / "error_memory_v0_13.json",
            "discourse": out / "discourse_state_v0_13.json",
            "world": out / "world_memory_v0_13.json",
            "session": out / "session_v0_13.json",
        }
        self.visual.learner.save(paths["visual"]); self.language.learner.save(paths["language"])
        self.concepts.save(paths["concepts"]); self.sync_graph(); self.graph.save(paths["graph"])
        self.errors.save(paths["errors"]); self.language.discourse.save(paths["discourse"])
        self.world.save(paths["world"])
        paths["session"].write_text(json.dumps({
            "version": self.VERSION,
            "seed": self.seed,
            "visual_test_history": self.visual_test_history,
            "language_test_history": self.language_test_history,
            "consolidation_history": self.consolidation_history,
            "world_test_history": self.world_test_history,
            "v012_bootstrap_experiences": self.v012_bootstrap_experiences,
            "memory_audit": self.memory_audit(),
        }, indent=2), encoding="utf-8")
        return {k: str(v) for k,v in paths.items()}
