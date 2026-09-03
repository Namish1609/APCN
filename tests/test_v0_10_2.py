import unittest
from pathlib import Path


class TestV0102(unittest.TestCase):
    def test_latest_launcher_targets_v102_ui(self):
        text = Path("run_desktop_v0_10.py").read_text(encoding="utf-8")
        self.assertIn("apcn_v10.ui_v102", text)

    def test_v102_training_layout_declares_all_three_activation_regions(self):
        text = Path("apcn_v10/ui_v102.py").read_text(encoding="utf-8")
        self.assertIn("PERCEPTION FIRING / SPIKING", text)
        self.assertIn("LANGUAGE FIRING / SPIKING", text)
        self.assertIn("DEFINITION FIRING / SPIKING", text)
        self.assertIn("QSplitter", text)


if __name__ == "__main__":
    unittest.main()
