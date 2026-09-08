from __future__ import annotations

from pathlib import Path
import json

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPlainTextEdit, QPushButton, QMessageBox

from apcn_v17.ui import APCNV17Window
from .session import CognitiveSessionV18


class APCNV18Window(APCNV17Window):
    """V0.18 studio: ordinary conversation routed through structured semantics."""

    def __init__(self, seed: int = 18):
        super().__init__(seed)
        self.seed = seed
        self.setWindowTitle("APCN V0.18 — Natural Semantic Conversation")

        out18 = Path("outputs/v0_18")
        out17 = Path("outputs/v0_17")
        if (out18 / "session_v0_18.json").exists():
            self.cognitive = CognitiveSessionV18.load_checkpoint(out18, seed=seed)
            note = "loaded V0.18 natural-conversation checkpoint"
        elif (out17 / "session_v0_17.json").exists():
            self.cognitive = CognitiveSessionV18.from_v17_checkpoint(out17, seed=seed)
            self.cognitive.save(out18)
            note = "imported V0.17 cognition and repaired language/knowledge bindings"
        else:
            self.cognitive = CognitiveSessionV18(seed)
            note = "new V0.18 natural-conversation memory"

        self.perception = self.cognitive.visual
        self.query_engine = self.cognitive.query
        self.visual_history = self.cognitive.visual_test_history
        self.consolidation_history = self.cognitive.consolidation_history

        self.p_hint.setText(
            "V0.18 fixes the conversation-routing failures exposed by desktop use. Ordinary questions such as "
            "'is Milo active?' and 'is Milo a creature?' now compile into the same structured semantic memory used by "
            "rules and proofs. 'X causes Y' is parsed as a causal operator, 'what do you know about X?' aggregates "
            "explicit semantic facts, and 'why?' uses the previous proof. Natural wording is produced by a bounded "
            "answer-plan realizer that has no access to truth/world/concept memory."
        )
        self.p_current.setText(f"V0.18: {note}")
        if hasattr(self, "c_train_status"):
            self.c_train_status.setText(
                "V0.18: do not random-train for these conversation cases. Test ordinary English; failures are classified "
                "as routing, semantics, reasoning, or realization."
            )
        self.tabs.addTab(self._build_v18_tab(), "Conversation Quality")
        self._refresh_v18()
        self._refresh_v17()
        self._refresh_v16()
        self._refresh_v15()
        self._refresh_header()

    def _build_v18_tab(self) -> QWidget:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(10, 10, 10, 10)
        title = QLabel("V0.18 — Unified Conversation → Semantics → Proof → English")
        title.setFont(QFont("Sans Serif", 14, 800))
        outer.addWidget(title)
        hint = QLabel(
            "The Conversation tab is the primary interface. This page shows the V0.18 architecture audit. "
            "The user transcript that exposed V0.17's failures is a development regression, not a blind benchmark."
        )
        hint.setWordWrap(True)
        outer.addWidget(hint)

        examples = QPlainTextEdit()
        examples.setReadOnly(True)
        examples.setMaximumHeight(270)
        examples.setPlainText(
            "Suggested desktop regression:\n\n"
            "fluxion means acceleration\n"
            "what is fluxion?\n"
            "I asked what is fluxation\n"
            "remember that milo is a cat\n"
            "remember that milo is active\n"
            "is milo active?\n"
            "is milo a cat?\n"
            "what do you know about milo?\n"
            "remember that every cat is a creature\n"
            "is milo a creature?\n"
            "why?\n"
            "remember that if battery is empty then device stops\n"
            "what follows if battery is empty?\n"
            "remember that power fails causes lamp turns off\n"
            "what causes lamp turns off?\n"
            "remember that door opens before light turns on\n"
            "what happens before light turns on?\n"
            "is zorbin a creature?"
        )
        outer.addWidget(examples)

        outer.addWidget(QLabel("V0.18 natural-conversation / realization audit"))
        self.v18_audit = QPlainTextEdit()
        self.v18_audit.setReadOnly(True)
        outer.addWidget(self.v18_audit, 1)
        save = QPushButton("Save V0.18 Checkpoint")
        save.clicked.connect(lambda: self._save_v15(False))
        outer.addWidget(save)
        return root

    def _refresh_v18(self):
        if not hasattr(self, "v18_audit") or not hasattr(self.cognitive, "v18_memory_audit"):
            return
        self.v18_audit.setPlainText(json.dumps(self.cognitive.v18_memory_audit(), indent=2, default=str))

    def _refresh_v15(self, last=None):
        if not hasattr(self.cognitive, "v18_memory_audit"):
            return super()._refresh_v15(last)
        if hasattr(self, "c_audit"):
            audit = {"v018": self.cognitive.v18_memory_audit(), "legacy_v015": self.cognitive.conversation_memory_audit()}
            if last is not None:
                audit = {"last_operation": last, **audit}
            self.c_audit.setPlainText(json.dumps(audit, indent=2, default=str))
        self._refresh_v18()

    def _save_v15(self, silent: bool = False):
        try:
            if hasattr(self.cognitive, "v18_memory_audit"):
                self.cognitive.save("outputs/v0_18")
                if not silent and hasattr(self, "c_status"):
                    self.c_status.setText("V0.18 checkpoint saved.")
                self._refresh_v18()
                return
            return super()._save_v15(silent)
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "Save failed", str(exc))

    def closeEvent(self, event):
        try:
            if hasattr(self, "_camera_stop"):
                self._camera_stop(silent=True)
            if hasattr(self.cognitive, "v18_memory_audit"):
                self.cognitive.save("outputs/v0_18")
            else:
                self.cognitive.save("outputs/v0_17")
        except Exception:
            pass
        event.accept()


def run_app(seed: int = 18) -> int:
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    win = APCNV18Window(seed)
    win.show()
    return app.exec()
