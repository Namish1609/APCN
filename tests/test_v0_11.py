import tempfile
import unittest
from pathlib import Path

import numpy as np

from apcn_v11.concept_graph import UnifiedConceptGraph
from apcn_v11.error_memory import ErrorMemory
from apcn_v11.consolidation import ConsolidationEngine
from apcn_v11.visual import PrototypeConceptLearner
from apcn_v11.language import SemanticLanguageLearnerV11
from apcn_v11.discourse import DiscourseEntityRegistry
from apcn_v11.language_teacher import RichSemanticTeacherV11
from apcn_v10.language_teacher import RichSemanticTeacher
from apcn_v10.semantic import SemanticNode
from apcn_v10.definitions import ConceptStore


class TestV011(unittest.TestCase):
    def test_error_memory_collapses_repeated_failures(self):
        mem = ErrorMemory(representative_limit=2)
        for i in range(100):
            mem.record("visual_shape", "rectangle", "ellipse", representative=f"case {i}")
        self.assertEqual(len(mem.signatures), 1)
        self.assertEqual(mem.top()[0].count, 100)
        self.assertEqual(sum(len(v) for v in mem.representatives.values()), 2)

    def test_error_memory_preserves_nested_program_path(self):
        class Failure:
            skill = "negation"
            utterance = "the circle is not below the square"
            expected = "NEGATE\n  ASSERT\n    R4(C0:S0:0, C1:S1:1)"
            predicted = "NEGATE\n  GOAL\n    R4(C0:S0:0, C1:S1:1)"
        class Report:
            failures = [Failure()]
        mem = ErrorMemory(); mem.record_language_report(Report())
        row = mem.top()[0]
        self.assertEqual(row.truth, "NEGATE>ASSERT")
        self.assertEqual(row.predicted, "NEGATE>GOAL")

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
        for i in range(900):
            intent = ("ASSERT", "QUERY", "GOAL")[i % 3]
            learner.observe(teacher.simple(intent=intent, held_out=False, skill="intent"))
        summary = learner.constructions.summary()
        self.assertGreater(summary["observations"], 800)
        self.assertGreater(summary["prefix_constructions"], 20)
        self.assertTrue(summary["strongest"])

    def test_unified_graph_bridges_shared_lexical_evidence(self):
        graph = UnifiedConceptGraph()
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

    def test_graph_sync_is_idempotent_view_not_learning_event(self):
        graph = UnifiedConceptGraph()
        class Stats:
            count = 10
        class Visual:
            token_stats = {"yellow": Stats()}
            def concept_quality(self, token): return .7
            def discover_families(self): return []
        graph.sync(visual=Visual())
        first = {(e.src, e.dst, e.relation): (e.weight, e.support) for e in graph.edges.values()}
        graph.sync(visual=Visual())
        second = {(e.src, e.dst, e.relation): (e.weight, e.support) for e in graph.edges.values()}
        self.assertEqual(first, second)

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

    def test_discourse_registry_tracks_focus_and_new_instance_identity(self):
        reg = DiscourseEntityRegistry()
        a = reg.new_entity("C4", "S3", role="subject", focus=True)
        b = reg.new_entity("C1", "S0", role="object")
        self.assertEqual(a.instance, 0)
        self.assertEqual(b.instance, 1)
        rel = SemanticNode.relation_node("R0", a, b, "GOAL")
        reg.ingest(rel)
        self.assertEqual(reg.resolve_reference().instance, 0)
        c = reg.resolve_description("C3", "S2", prefer_existing=True)
        self.assertIsNotNone(c)
        self.assertEqual(c.instance, 2)
        self.assertEqual(reg.summary()["entity_count"], 3)

    def test_discourse_registry_persistence(self):
        reg = DiscourseEntityRegistry(); a = reg.new_entity("C2", "S3", focus=True); reg.new_entity("C1", "S0")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "discourse.json"; reg.save(p); loaded = DiscourseEntityRegistry.load(p)
        self.assertEqual(loaded.focus.instance, a.instance)
        self.assertEqual(loaded.summary()["entity_count"], 2)

    def test_reference_teacher_avoids_ambiguous_duplicate_target(self):
        teacher = RichSemanticTeacherV11(seed=1101)
        for _ in range(40):
            first, second = teacher.reference_pair(held_out=True)
            a = first.program.atom(); b = second.program.atom()
            self.assertIsNotNone(a); self.assertIsNotNone(b)
            previous = {(a.subject.color, a.subject.shape), (a.object.color, a.object.shape)}
            self.assertNotIn((b.object.color, b.object.shape), previous)
            self.assertEqual(b.object.instance, 2)

    def test_graph_persistence(self):
        g = UnifiedConceptGraph(); g.upsert_node("concept:x", "x", "concept", confidence=.7); g.upsert_node("concept:y", "y", "concept"); g.strengthen("concept:x", "concept:y", "DEPENDS_ON", .9, 2)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "graph.json"; g.save(p); r = UnifiedConceptGraph.load(p)
        self.assertIn("concept:x", r.nodes)
        self.assertEqual(len(r.neighbors("concept:x", "DEPENDS_ON")), 1)

    def test_v011_ui_contains_consolidation_and_migration_controls(self):
        text = Path("apcn_v11/ui.py").read_text(encoding="utf-8")
        self.assertIn("Run 1 Automatic Consolidation Cycle", text)
        self.assertIn("Migrate V0.10 Memory", text)
        self.assertIn("Unified concept graph", text)
        self.assertIn("Memory + discourse audit", text)

    def test_v011_release_metadata_and_launcher(self):
        # This is a historical-version regression. Newer releases are allowed to
        # advance the repository VERSION/README; V0.11's own artifacts must remain.
        version = Path("VERSION").read_text(encoding="utf-8").splitlines()[0]
        launcher = Path("run_desktop_v0_11.py").read_text(encoding="utf-8")
        legacy_readme = Path("README_V0_11.md").read_text(encoding="utf-8")
        major, minor, patch = (int(x) for x in version.split(".")[:3])
        self.assertGreaterEqual((major, minor, patch), (0, 11, 0))
        self.assertIn("from apcn_v11.ui import run_app", launcher)
        self.assertIn("APCN V0.11", legacy_readme)
        self.assertIn("Migrate V0.10 Memory", legacy_readme)


if __name__ == "__main__":
    unittest.main()
