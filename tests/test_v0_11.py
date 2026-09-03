import tempfile
import unittest
from pathlib import Path

import numpy as np

from apcn_v11.concept_graph import UnifiedConceptGraph
from apcn_v11.error_memory import ErrorMemory
from apcn_v11.consolidation import ConsolidationEngine
from apcn_v11.visual import PrototypeConceptLearner
from apcn_v11.language import SemanticLanguageLearnerV11
from apcn_v10.language_teacher import RichSemanticTeacher
from apcn_v10.definitions import ConceptStore


class TestV011(unittest.TestCase):
    def test_error_memory_collapses_repeated_failures(self):
        mem = ErrorMemory(representative_limit=2)
        for i in range(100):
            mem.record("visual_shape", "rectangle", "ellipse", representative=f"case {i}")
        self.assertEqual(len(mem.signatures), 1)
        self.assertEqual(mem.top()[0].count, 100)
        self.assertEqual(sum(len(v) for v in mem.representatives.values()), 2)

    def test_visual_prototypes_are_bounded_not_episode_archive(self):
        learner = PrototypeConceptLearner(max_prototypes=4, new_prototype_distance=.15)
        rng = np.random.default_rng(11)
        for _ in range(80):
            learner.observe("this is a zorp", rng.normal(0, 1, learner.sensor.dim))
        self.assertEqual(learner.episode_count, 80)
        self.assertLessEqual(len(learner.prototype_banks["zorp"]), 4)
        self.assertFalse(hasattr(learner, "episodes"))

    def test_construction_inducer_learns_abstract_patterns(self):
        learner = SemanticLanguageLearnerV11()
        teacher = RichSemanticTeacher(seed=111)
        # Balanced experiences establish lexical grounding and recurring
        # sentence constructions without hardcoding intent phrases in learner.
        for i in range(900):
            intent = ("ASSERT", "QUERY", "GOAL")[i % 3]
            learner.observe(teacher.simple(intent=intent, held_out=False, skill="intent"))
        summary = learner.constructions.summary()
        self.assertGreater(summary["observations"], 800)
        self.assertGreater(summary["prefix_constructions"], 20)
        self.assertTrue(summary["strongest"])

    def test_unified_graph_bridges_shared_lexical_evidence(self):
        graph = UnifiedConceptGraph()
        # Small fake interfaces keep this test about graph semantics.
        class Stats:
            count = 20
        class Visual:
            token_stats = {"yellow": Stats()}
            def concept_quality(self, token): return .8
            def discover_families(self): return []
        class Language:
            feature_totals = {"color:C0": 20}
            cue_totals = {"yellow": 20}
            def cue_support(self, cue, feature): return 20 if cue == "yellow" and feature == "color:C0" else 0
            def feature_purity(self, cue, feature): return 1.0 if self.cue_support(cue, feature) else 0.0
            def cue_score(self, cue, feature, position=None): return .9 if self.cue_support(cue, feature) else 0.0
        graph.sync(visual=Visual(), language=Language())
        bridges = [e for e in graph.edges.values() if e.relation == "SAME_CONCEPT_HYPOTHESIS"]
        self.assertTrue(bridges)
        self.assertGreater(bridges[0].weight, .7)

    def test_definition_dependencies_enter_unified_graph(self):
        store = ConceptStore()
        store.add_primitive("mass")
        store.add_primitive("volume")
        store.learn_definition("density is mass divided by volume")
        graph = UnifiedConceptGraph(); graph.sync(definitions=store)
        deps = graph.neighbors("concept:density", "DEPENDS_ON")
        self.assertEqual({e.dst for e in deps}, {"concept:mass", "concept:volume"})

    def test_consolidation_prioritizes_repeated_confusion(self):
        mem = ErrorMemory()
        for _ in range(30): mem.record("visual_shape", "rectangle", "ellipse")
        for _ in range(3): mem.record("visual_shape", "square", "circle")
        engine = ConsolidationEngine(mem)
        rows = engine.prescriptions(limit=3)
        self.assertEqual(rows[0].target, "rectangle")
        self.assertEqual(rows[0].contrast, "ellipse")

    def test_graph_persistence(self):
        g = UnifiedConceptGraph(); g.upsert_node("concept:x", "x", "concept", confidence=.7); g.upsert_node("concept:y", "y", "concept"); g.strengthen("concept:x", "concept:y", "DEPENDS_ON", .9, 2)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "graph.json"; g.save(p); r = UnifiedConceptGraph.load(p)
        self.assertIn("concept:x", r.nodes)
        self.assertEqual(len(r.neighbors("concept:x", "DEPENDS_ON")), 1)


if __name__ == "__main__":
    unittest.main()
