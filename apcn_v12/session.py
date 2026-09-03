from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import json

from apcn_v08.session import TrainingSessionV08
from apcn_v10.definitions import ConceptStore, DefinitionCurriculum
from apcn_v10.query import KnowledgeQueryEngine
from apcn_v11.session import CognitiveSessionV11
from apcn_v11.language import AdaptiveLanguageSessionV11, SemanticLanguageLearnerV11
from apcn_v11.error_memory import ErrorMemory
from apcn_v11.discourse import DiscourseEntityRegistry
from apcn_v11.consolidation import ConsolidationEngine
from apcn_v11.concept_graph import UnifiedConceptGraph

from .visual import SelfOrganizingVisualLearner


class TrainingSessionV12(TrainingSessionV08):
    def __init__(self, seed: int = 12, learner: Optional[SelfOrganizingVisualLearner] = None):
        super().__init__(seed=seed, learner=learner or SelfOrganizingVisualLearner())


class CognitiveSessionV12(CognitiveSessionV11):
    VERSION = "0.12.0"

    def __init__(self, seed: int = 12):
        super().__init__(seed)
        self.visual = TrainingSessionV12(seed)
        self.graph = UnifiedConceptGraph()
        self.consolidation = ConsolidationEngine(self.errors)
        self.visual_test_history = []
        self.test_history = self.language_test_history
        self.v012_bootstrap_experiences = 0

    @staticmethod
    def _read_episode_count(path: Path) -> int:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return int(data.get("episode_count", 0))
        except Exception:
            return 0

    @classmethod
    def from_v11_checkpoint(cls, output_dir: str | Path = "outputs/v0_11", *, seed: int = 12,
                            representation_bootstrap: int = 240) -> "CognitiveSessionV12":
        """Migrate knowledge that is representation-compatible.

        Language/definitions/error/discourse memory transfers directly. The old
        23-D visual distributions do not: their dimensional meaning is different.
        Their episode count and error signatures are retained as provenance and
        curriculum evidence, while V0.12 forms fresh compact visual statistics.
        """
        out = Path(output_dir)
        obj = cls(seed)

        old_visual = out / "visual_memory_v0_11.json"
        if old_visual.exists():
            obj.visual.learner.legacy_visual_episode_count = obj._read_episode_count(old_visual)

        lp = out / "language_memory_v0_11.json"
        if lp.exists():
            obj.language = AdaptiveLanguageSessionV11(seed, learner=SemanticLanguageLearnerV11.load(lp))

        cp = out / "concept_store_v0_11.json"
        if cp.exists():
            obj.concepts = ConceptStore.load(cp)
            obj.definitions = DefinitionCurriculum(obj.concepts)
            obj.query = KnowledgeQueryEngine(obj.concepts)

        ep = out / "error_memory_v0_11.json"
        if ep.exists():
            obj.errors = ErrorMemory.load(ep)
            obj.consolidation = ConsolidationEngine(obj.errors)

        dp = out / "discourse_state_v0_11.json"
        if dp.exists():
            obj.language.discourse = DiscourseEntityRegistry.load(dp)

        sp = out / "session_v0_11.json"
        if sp.exists():
            try:
                state = json.loads(sp.read_text(encoding="utf-8"))
                obj.language_test_history = list(state.get("language_test_history", []))
                obj.test_history = obj.language_test_history
                obj.consolidation_history = list(state.get("consolidation_history", []))
            except Exception:
                pass

        if representation_bootstrap > 0:
            obj.v012_bootstrap_experiences = obj.visual.learner.bootstrap_representation(
                obj.visual.teacher, representation_bootstrap, difficulty=.72)
        obj.sync_graph()
        return obj

    @classmethod
    def load_checkpoint(cls, output_dir: str | Path = "outputs/v0_12", *, seed: int = 12) -> "CognitiveSessionV12":
        out = Path(output_dir)
        obj = cls(seed)
        vp = out / "visual_memory_v0_12.json"
        if vp.exists():
            obj.visual = TrainingSessionV12(seed, learner=SelfOrganizingVisualLearner.load(vp))
        lp = out / "language_memory_v0_12.json"
        if lp.exists():
            obj.language = AdaptiveLanguageSessionV11(seed, learner=SemanticLanguageLearnerV11.load(lp))
        cp = out / "concept_store_v0_12.json"
        if cp.exists():
            obj.concepts = ConceptStore.load(cp)
            obj.definitions = DefinitionCurriculum(obj.concepts)
            obj.query = KnowledgeQueryEngine(obj.concepts)
        ep = out / "error_memory_v0_12.json"
        if ep.exists():
            obj.errors = ErrorMemory.load(ep); obj.consolidation = ConsolidationEngine(obj.errors)
        dp = out / "discourse_state_v0_12.json"
        if dp.exists():
            obj.language.discourse = DiscourseEntityRegistry.load(dp)
        gp = out / "unified_concept_graph_v0_12.json"
        if gp.exists():
            try: obj.graph = UnifiedConceptGraph.load(gp)
            except Exception: obj.graph = UnifiedConceptGraph()
        sp = out / "session_v0_12.json"
        if sp.exists():
            state = json.loads(sp.read_text(encoding="utf-8"))
            obj.visual_test_history = list(state.get("visual_test_history", []))
            obj.language_test_history = list(state.get("language_test_history", []))
            obj.test_history = obj.language_test_history
            obj.consolidation_history = list(state.get("consolidation_history", []))
            obj.v012_bootstrap_experiences = int(state.get("v012_bootstrap_experiences", 0))
        obj.sync_graph()
        return obj

    def visual_coverage(self) -> Dict[str, object]:
        learner = self.visual.learner
        colors = list(self.visual.teacher.color_words)
        shapes = list(self.visual.teacher.shape_words)
        missing_colors = [x for x in colors if learner.token_stats.get(x) is None or learner.token_stats[x].count < 3]
        missing_shapes = [x for x in shapes if learner.token_stats.get(x) is None or learner.token_stats[x].count < 3]
        return {
            "missing_colors": missing_colors,
            "missing_shapes": missing_shapes,
            "complete": not missing_colors and not missing_shapes,
        }

    def _balanced_visual_bootstrap(self, experiences: int) -> int:
        colors = list(self.visual.teacher.color_words); shapes = list(self.visual.teacher.shape_words)
        n = max(0, int(experiences)); done = 0
        for i in range(n):
            color = colors[i % len(colors)]
            shape = shapes[(i // len(colors)) % len(shapes)]
            d = .14 + .66 * ((i % 120) / 119.0)
            ep = self.visual.teacher.generate(color=color, shape=shape, difficulty=d,
                                              add_distractors=d >= .42)
            self.visual.learner.train_episode(ep); done += 1
        self.visual.curriculum.index = self.visual.learner.episode_count
        return done

    def consolidate_visual(self, experiences: int = 500) -> Dict[str, int]:
        """Bootstrap the new representation when required, then target errors."""
        requested = max(0, int(experiences)); trained = 0
        coverage = self.visual_coverage()
        if not coverage["complete"] and requested > 0:
            # Enough balanced examples to expose every factor repeatedly without
            # requiring the user to decide which color/shape to train first.
            bootstrap_n = min(requested, max(180, requested // 2))
            trained += self._balanced_visual_bootstrap(bootstrap_n)
        remaining = max(0, requested - trained)
        if remaining > 0:
            result = super().consolidate_visual(remaining)
            trained += int(result.get("trained", 0))
            targets = int(result.get("targets", 0))
        else:
            targets = 0
        return {"trained": trained, "targets": targets,
                "balanced_bootstrap": min(trained, requested-remaining)}

    def memory_audit(self) -> Dict[str, object]:
        base = super().memory_audit()
        base["v012_representation"] = self.visual.learner.representation_summary()
        base["v012_visual_coverage"] = self.visual_coverage()
        base["v012_representation_bootstrap"] = self.v012_bootstrap_experiences
        return base

    def save(self, output_dir: str | Path = "outputs/v0_12") -> Dict[str, str]:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        visual = out / "visual_memory_v0_12.json"
        language = out / "language_memory_v0_12.json"
        concepts = out / "concept_store_v0_12.json"
        graph = out / "unified_concept_graph_v0_12.json"
        errors = out / "error_memory_v0_12.json"
        discourse = out / "discourse_state_v0_12.json"
        state = out / "session_v0_12.json"

        self.visual.learner.save(visual)
        self.language.learner.save(language)
        self.concepts.save(concepts)
        self.sync_graph(); self.graph.save(graph)
        self.errors.save(errors)
        self.language.discourse.save(discourse)
        state.write_text(json.dumps({
            "version": self.VERSION,
            "seed": self.seed,
            "visual_test_history": self.visual_test_history,
            "language_test_history": self.language_test_history,
            "consolidation_history": self.consolidation_history,
            "v012_bootstrap_experiences": self.v012_bootstrap_experiences,
            "memory_audit": self.memory_audit(),
        }, indent=2), encoding="utf-8")
        return {"visual": str(visual), "language": str(language), "concepts": str(concepts),
                "graph": str(graph), "errors": str(errors), "discourse": str(discourse),
                "session": str(state)}
