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
from .language import AdaptiveLanguageSessionV11, SemanticLanguageLearnerV11
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

    @classmethod
    def from_memories(cls, *, seed: int = 11, visual_memory: str | Path | None = None,
                      language_memory: str | Path | None = None,
                      concept_memory: str | Path | None = None) -> "CognitiveSessionV11":
        """Migrate existing V0.8/V0.10 compact memories instead of retraining from zero."""
        obj = cls(seed)
        if visual_memory is not None and Path(visual_memory).exists():
            learner = PrototypeConceptLearner.load(visual_memory)
            obj.visual = TrainingSessionV11(seed, learner=learner)
        if language_memory is not None and Path(language_memory).exists():
            learner = SemanticLanguageLearnerV11.load(language_memory)
            obj.language = AdaptiveLanguageSessionV11(seed, learner=learner)
        if concept_memory is not None and Path(concept_memory).exists():
            obj.concepts = ConceptStore.load(concept_memory)
            obj.definitions.store = obj.concepts
            obj.definitions.index = 0
            obj.query = KnowledgeQueryEngine(obj.concepts)
        return obj

    def train_visual(self, experiences: int) -> None:
        for _ in range(max(0, int(experiences))):
            self.visual.step()

    def train_language(self, experiences: int) -> None:
        remaining = max(0, int(experiences))
        while remaining > 0:
            before = self.language.learner.episode_count
            self.language.step()
            gained = max(1, self.language.learner.episode_count - before)
            remaining -= gained

    def learn_definition_curriculum(self) -> None:
        self.definitions.train_all_once()
        self.query = KnowledgeQueryEngine(self.concepts)

    def test_visual(self, samples: int = 500, difficulty: float = .82):
        rep = run_bulk_test(self.visual.learner, samples, difficulty,
                            seed=self.seed + 31001 + len(self.visual_test_history)*31)
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

    def consolidate_visual(self, experiences: int = 400) -> Dict[str, int]:
        """Generate targeted minimal contrasts from current visual confusions."""
        rows = [p for p in self.prescriptions(20) if p.domain in {"visual_shape", "visual_color"}]
        if not rows:
            rows = self.consolidation.prescriptions(
                visual_learner=self.visual.learner,
                colors=self.visual.teacher.color_words,
                shapes=self.visual.teacher.shape_words,
                limit=12,
            )
            rows = [p for p in rows if p.domain in {"visual_shape", "visual_color"}]
        if not rows:
            return {"trained": 0, "targets": 0}

        colors = list(self.visual.teacher.color_words)
        shapes = list(self.visual.teacher.shape_words)
        trained = 0
        n = max(0, int(experiences))
        for i in range(n):
            p = rows[i % len(rows)]
            difficulty = 0.24 + 0.64 * ((i % 80) / 79.0)
            if p.domain == "visual_shape":
                shape = p.target if (i // len(rows)) % 2 == 0 else p.contrast
                if shape not in shapes:
                    continue
                color = colors[(i // 2) % len(colors)]
            else:
                color = p.target if (i // len(rows)) % 2 == 0 else p.contrast
                if color not in colors:
                    continue
                shape = shapes[(i // 2) % len(shapes)]
            ep = self.visual.teacher.generate(color=color, shape=shape, difficulty=difficulty,
                                              add_distractors=difficulty >= .45)
            self.visual.learner.train_episode(ep)
            trained += 1
        self.visual.curriculum.index = self.visual.learner.episode_count
        return {"trained": trained, "targets": len(rows)}

    def _observe_language_episodes(self, skill: str, episodes) -> int:
        learned = 0
        discourse_focus = None
        for ep in episodes:
            focus = ep.discourse_focus or discourse_focus
            pred = self.language.learner.parse(ep.utterance, focus)
            ok = self.language._skill_correct(skill, pred, ep.program)
            if skill in self.language.skills:
                self.language.skills[skill].update(ok)
            self.language.learner.observe(ep)
            atom = ep.program.atom()
            if atom is not None and atom.subject is not None:
                discourse_focus = atom.subject
            learned += 1
        self.language.last_skill = skill
        return learned

    def consolidate_language(self, experiences: int = 500) -> Dict[str, int]:
        """Target recurring program/intent failures with fresh contrastive language."""
        errors = self.errors.top("language_program", limit=20)
        trained = 0
        requested = max(0, int(experiences))
        for i in range(requested):
            e = errors[i % len(errors)] if errors else None
            if e is None:
                before = self.language.learner.episode_count
                self.language.step()
                trained += max(1, self.language.learner.episode_count - before)
                continue

            skill = e.context if e.context in self.language.SKILLS else "intent"
            if skill == "intent" and e.truth in {"ASSERT", "QUERY", "GOAL"}:
                episodes = [self.language.teacher.simple(intent=e.truth, held_out=False, skill="intent")]
            elif e.truth == "NEGATE":
                skill, episodes = "negation", [self.language.teacher.negated(False)]
            elif e.truth == "GROUP":
                skill, episodes = "group", [self.language.teacher.group(False)]
            elif e.truth == "SEQUENCE":
                skill, episodes = "sequence", [self.language.teacher.sequence(False)]
            else:
                episodes = self.language.teacher.for_skill(skill, held_out=False)
            trained += self._observe_language_episodes(skill, episodes)
        return {"trained": trained, "targets": len(errors)}

    def consolidation_cycle(self, *, visual_test: int = 300, language_test: int = 360,
                            visual_train: int = 400, language_train: int = 500,
                            difficulty: float = .82) -> Dict[str, object]:
        """Read-only diagnose -> targeted new evidence -> read-only retest."""
        v0 = self.test_visual(visual_test, difficulty)
        l0 = self.test_language(language_test)
        before = {
            "visual_joint": v0.joint_accuracy,
            "visual_shape": v0.shape_accuracy,
            "language_exact": l0.exact_accuracy,
            "language_intent": l0.intent_accuracy,
        }
        planned = [p.__dict__ for p in self.prescriptions(12)]
        vtrain = self.consolidate_visual(visual_train)
        ltrain = self.consolidate_language(language_train)
        v1 = self.test_visual(visual_test, difficulty)
        l1 = self.test_language(language_test)
        after = {
            "visual_joint": v1.joint_accuracy,
            "visual_shape": v1.shape_accuracy,
            "language_exact": l1.exact_accuracy,
            "language_intent": l1.intent_accuracy,
        }
        self.sync_graph()
        return {"before": before, "after": after, "visual_training": vtrain,
                "language_training": ltrain, "prescriptions": planned}

    def memory_audit(self) -> Dict[str, object]:
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
