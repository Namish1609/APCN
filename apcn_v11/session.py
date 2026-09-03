from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import json

from apcn_v08.session import TrainingSessionV08
from apcn_v08.testing_v082 import run_bulk_test
from apcn_v10.definitions import ConceptStore, DefinitionCurriculum
from apcn_v10.language_session import run_generated_language_test
from apcn_v10.query import KnowledgeQueryEngine

from .visual import PrototypeConceptLearner
from .language import AdaptiveLanguageSessionV11
from .concept_graph import UnifiedConceptGraph
from .error_memory import ErrorMemory
from .consolidation import ConsolidationEngine


class TrainingSessionV11(TrainingSessionV08):
    def __init__(self, seed: int = 11, learner: Optional[PrototypeConceptLearner] = None):
        super().__init__(seed=seed, learner=learner or PrototypeConceptLearner())


class CognitiveSessionV11:
    VERSION = "0.11.0"

    def __init__(self, seed: int = 11):
        self.seed = int(seed)
        self.visual = TrainingSessionV11(seed)
        self.language = AdaptiveLanguageSessionV11(seed)
        self.concepts = ConceptStore()
        self.definitions = DefinitionCurriculum(self.concepts)
        self.graph = UnifiedConceptGraph()
        self.errors = ErrorMemory()
        self.consolidation = ConsolidationEngine(self.errors)
        self.query = KnowledgeQueryEngine(self.concepts)
        self.visual_test_history = []
        self.language_test_history = []

    def train_visual(self, experiences: int) -> None:
        for _ in range(max(0, int(experiences))):
            self.visual.step()

    def train_language(self, experiences: int) -> None:
        while experiences > 0:
            before = self.language.learner.episode_count
            self.language.step()
            gained = max(1, self.language.learner.episode_count - before)
            experiences -= gained

    def learn_definition_curriculum(self) -> None:
        self.definitions.train_all_once()
        self.query = KnowledgeQueryEngine(self.concepts)

    def test_visual(self, samples: int = 500, difficulty: float = .82):
        rep = run_bulk_test(self.visual.learner, samples, difficulty, seed=self.seed + 31001 + len(self.visual_test_history)*31)
        self.errors.record_visual_report(rep)
        self.visual_test_history.append({
            "episodes": self.visual.learner.episode_count,
            "color": rep.color_accuracy,
            "shape": rep.shape_accuracy,
            "joint": rep.joint_accuracy,
        })
        return rep

    def test_language(self, samples: int = 600):
        rep = run_generated_language_test(self.language.learner, samples=samples,
                                          seed=self.seed + 41001 + len(self.language_test_history)*37)
        self.errors.record_language_report(rep)
        self.language_test_history.append({
            "episodes": self.language.learner.episode_count,
            "exact": rep.exact_accuracy,
            "intent": rep.intent_accuracy,
            "relation": rep.relation_accuracy,
            "operator": rep.operator_accuracy,
        })
        return rep

    def sync_graph(self) -> Dict[str, object]:
        self.graph.sync(visual=self.visual.learner, language=self.language.learner, definitions=self.concepts)
        return self.graph.summary()

    def prescriptions(self, limit: int = 12):
        return self.consolidation.prescriptions(
            visual_learner=self.visual.learner,
            colors=self.visual.teacher.color_words,
            shapes=self.visual.teacher.shape_words,
            language_learner=self.language.learner,
            limit=limit,
        )

    def memory_audit(self) -> Dict[str, object]:
        # These are logical counts; disk bytes are reported after save by caller.
        return {
            "visual_episodes_seen": self.visual.learner.episode_count,
            "visual_raw_examples_retained": 0,
            "visual_token_stats": len(self.visual.learner.token_stats),
            "visual_prototypes": self.visual.learner.prototype_summary(),
            "language_episodes_seen": self.language.learner.episode_count,
            "language_raw_sentences_retained": 0,
            "language_unique_cues": len(self.language.learner.cue_totals),
            "language_constructions": self.language.learner.constructions.summary(),
            "definition_records": len(self.concepts.records),
            "error_memory": self.errors.summary(),
            "concept_graph": self.graph.summary(),
        }

    def save(self, output_dir: str | Path = "outputs/v0_11") -> Dict[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        visual = out / "visual_memory_v0_11.json"
        language = out / "language_memory_v0_11.json"
        concepts = out / "concept_store_v0_11.json"
        graph = out / "unified_concept_graph_v0_11.json"
        errors = out / "error_memory_v0_11.json"
        state = out / "session_v0_11.json"
        self.visual.learner.save(visual)
        self.language.learner.save(language)
        self.concepts.save(concepts)
        self.graph.save(graph)
        self.errors.save(errors)
        state.write_text(json.dumps({
            "version": self.VERSION,
            "seed": self.seed,
            "visual_test_history": self.visual_test_history,
            "language_test_history": self.language_test_history,
            "memory_audit": self.memory_audit(),
        }, indent=2), encoding="utf-8")
        return {"visual": str(visual), "language": str(language), "concepts": str(concepts),
                "graph": str(graph), "errors": str(errors), "session": str(state)}
