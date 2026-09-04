import tempfile
import unittest
from pathlib import Path

import numpy as np

from apcn_v13.identity import BoundedAppearanceBank, InstanceMemory
from apcn_v13.world import Detection, PersistentWorldModel
from apcn_v13.session import CognitiveSessionV13


class TestV013(unittest.TestCase):
    def test_appearance_memory_is_bounded_and_not_episode_archive(self):
        bank = BoundedAppearanceBank(max_prototypes=4, novelty_threshold=.05)
        rng = np.random.default_rng(13)
        for _ in range(200):
            bank.observe(rng.normal(0, 1, size=32))
        self.assertLessEqual(len(bank.prototypes), 4)
        self.assertEqual(bank.observations, 200)
        d = bank.to_dict()
        self.assertEqual(d["raw_frames_retained"], 0)
        self.assertEqual(d["raw_descriptors_retained"], 0)

    def test_human_correction_adds_negative_and_positive_evidence(self):
        mem = InstanceMemory(max_views=4, strong_threshold=.35, probable_threshold=.2)
        a = np.zeros(24, dtype=float); b = np.ones(24, dtype=float) * .2
        ia = mem.teach("object_a", a)
        ib = mem.teach("object_b", b)
        probe = b.copy()
        before_a = ia.appearance_score(probe)
        mem.correct(wrong_instance_id=ia.instance_id, correct_name="object_b", descriptor=probe)
        self.assertGreater(len(ia.negative.prototypes), 0)
        self.assertGreater(ib.positive.observations, 1)
        self.assertLess(ia.appearance_score(probe), before_a)
        self.assertGreater(ib.appearance_score(probe), ia.appearance_score(probe))

    def test_world_uses_explicit_occlusion_and_reappearance_states(self):
        mem = InstanceMemory(max_views=4, strong_threshold=.2, probable_threshold=.1, ambiguity_margin=.0)
        x = np.linspace(0,1,40)
        inst = mem.teach("my_object", x, category=("thing",))
        world = PersistentWorldModel(mem, lost_after=8, out_of_view_after=4)
        box = (.30,.40,.12,.12)
        world.teach_instance("my_object", x, box, timestamp=0.0, category=("thing",))
        rows = world.process_frame([Detection(x.copy(), (.38,.40,.12,.12), ("thing",))], timestamp=1.0)
        self.assertEqual(rows[0].instance_id, inst.instance_id)
        world.process_frame([], timestamp=2.0, occluders=[(.35,.30,.40,.40)])
        self.assertEqual(world.tracks[inst.instance_id].state, "OCCLUDED")
        rows = world.process_frame([Detection(x.copy(), (.48,.40,.12,.12), ("thing",))], timestamp=3.0)
        self.assertEqual(rows[0].instance_id, inst.instance_id)
        self.assertEqual(world.tracks[inst.instance_id].state, "VISIBLE")
        self.assertTrue(any(e.kind == "REAPPEARED" for e in world.events))

    def test_where_answer_never_uses_simulator_truth(self):
        mem = InstanceMemory(max_views=3, strong_threshold=.2, probable_threshold=.1, ambiguity_margin=.0)
        x = np.arange(30,dtype=float)/30.0
        world = PersistentWorldModel(mem)
        iid = world.teach_instance("keys", x, (.72,.68,.10,.10), timestamp=1.0, category=("keys",))
        row = world.where("keys")
        self.assertTrue(row["known"])
        self.assertEqual(row["instance_id"], iid)
        self.assertIn("visible", row["answer"].lower())
        world.process_frame([], timestamp=2.0, occluders=[(.65,.60,.30,.30)])
        row = world.where("keys")
        self.assertIn(row["state"], {"OCCLUDED","OUT_OF_VIEW"})
        self.assertNotIn("ground truth", row["answer"].lower())

    def test_world_checkpoint_persists_belief_not_working_frame(self):
        s = CognitiveSessionV13(seed=13)
        x = np.zeros(s.visual.learner.sensor.dim, dtype=float)
        s.world.teach_instance("wallet", x, (.2,.2,.2,.2), timestamp=2.0, category=("wallet",))
        with tempfile.TemporaryDirectory() as td:
            s.save(td)
            restored = CognitiveSessionV13.load_checkpoint(td, seed=13)
            self.assertIsNotNone(restored.world.instances.by_name("wallet"))
            self.assertIsNone(restored._last_descriptor)
            self.assertEqual(restored.world.memory_summary()["raw_video_frames_retained"], 0)

    def test_v013_ui_and_release_surface_exist(self):
        text = Path("apcn_v13/ui.py").read_text(encoding="utf-8")
        self.assertIn("World Memory", text)
        self.assertIn("Teach Named Instance", text)
        self.assertIn("Observe / Match", text)
        self.assertIn("Correct Current Observation To Name", text)
        self.assertIn("Where is it?", text)


if __name__ == "__main__":
    unittest.main()
