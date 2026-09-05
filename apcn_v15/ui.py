from __future__ import annotations

from pathlib import Path
import json

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QLineEdit, QSpinBox, QPlainTextEdit, QGroupBox, QMessageBox, QSplitter,
)
from PyQt6.QtCore import Qt

from apcn_v14.launch_window import APCNV14LaunchWindow
from .session import CognitiveSessionV15


class APCNV15Window(APCNV14LaunchWindow):
    """Language-only V0.15 studio with an explicit conversational interface."""

    def __init__(self, seed: int = 15):
        super().__init__(seed)
        self.seed = seed
        self.setWindowTitle("APCN V0.15 — Conversational Language Studio")

        out15 = Path("outputs/v0_15")
        out14 = Path("outputs/v0_14")
        if (out15 / "session_v0_15.json").exists():
            self.cognitive = CognitiveSessionV15.load_checkpoint(out15, seed=seed)
            note = "loaded V0.15 conversational checkpoint"
        elif (out14 / "session_v0_14.json").exists():
            self.cognitive = CognitiveSessionV15.from_v14_checkpoint(out14, seed=seed)
            self.cognitive.save(out15)
            note = "imported V0.14 cognition; added conversational English memory"
        else:
            self.cognitive = CognitiveSessionV15(seed)
            note = "new V0.15 conversational memory"

        self.perception = self.cognitive.visual
        self.query_engine = self.cognitive.query
        self.visual_history = self.cognitive.visual_test_history
        self.consolidation_history = self.cognitive.consolidation_history

        # V0.15 has no face-development surface. The underlying V0.14 memory is
        # preserved for backward compatibility, but this research release spends
        # its active development/training budget entirely on language.
        for i in reversed(range(self.tabs.count())):
            if self.tabs.tabText(i) == "Self Face Camera":
                self.tabs.removeTab(i)

        self.p_hint.setText(
            "V0.15 is 100% language-focused. Existing perception/world memory remains available as grounding, "
            "but automatic V0.15 training adds zero visual experiences. The new Conversation tab talks over explicit "
            "APCN knowledge, learns aliases/definitions/facts, maintains semantic dialogue context and asks for clarification when language is unknown."
        )
        self.p_current.setText(f"V0.15: {note}")
        self.tabs.addTab(self._build_conversation_tab(), "Conversation")
        self._refresh_v15()
        self._refresh_header()

    def _build_conversation_tab(self) -> QWidget:
        root = QWidget(); outer = QVBoxLayout(root); outer.setContentsMargins(10,10,10,10)
        title = QLabel("Conversational English — Explicit Memory, No Hidden LLM")
        title.setFont(QFont("Sans Serif", 14, 800)); outer.addWidget(title)
        hint = QLabel(
            "Talk normally within APCN's current knowledge. Follow-ups such as 'what does it depend on?', 'why?', and 'tell me more' use semantic dialogue state. "
            "Teach vocabulary with 'fluxion means acceleration', definitions with 'speed is distance divided by time', and simple facts with 'remember that orbix is a sensor'. "
            "Unknown constructions are surfaced as unknown instead of guessed."
        )
        hint.setWordWrap(True); outer.addWidget(hint)

        split = QSplitter(Qt.Orientation.Horizontal); outer.addWidget(split, 1)
        left = QWidget(); ll = QVBoxLayout(left); split.addWidget(left)
        right = QWidget(); rl = QVBoxLayout(right); split.addWidget(right)
        split.setSizes([760, 500])

        self.c_chat = QPlainTextEdit(); self.c_chat.setReadOnly(True)
        self.c_chat.setPlaceholderText("Conversation appears here. The transcript is UI working state and is not persisted as long-term raw chat memory.")
        ll.addWidget(self.c_chat, 1)
        sendrow = QHBoxLayout(); self.c_input = QLineEdit(); self.c_input.setPlaceholderText("Talk to APCN...")
        self.c_input.returnPressed.connect(self._v15_send)
        send = QPushButton("Send"); send.clicked.connect(self._v15_send)
        clear = QPushButton("Clear View"); clear.clicked.connect(self.c_chat.clear)
        sendrow.addWidget(self.c_input, 1); sendrow.addWidget(send); sendrow.addWidget(clear); ll.addLayout(sendrow)
        self.c_status = QLabel("Ready."); self.c_status.setWordWrap(True); ll.addWidget(self.c_status)

        training = QGroupBox("100% language training / testing"); grid = QGridLayout(training); rl.addWidget(training)
        self.c_steps = QSpinBox(); self.c_steps.setRange(20, 50000); self.c_steps.setSingleStep(500); self.c_steps.setValue(2000)
        grid.addWidget(QLabel("Language experiences"),0,0); grid.addWidget(self.c_steps,0,1)
        train = QPushButton("Train Language Only"); train.clicked.connect(self._v15_train)
        test = QPushButton("Held-Out Language Test"); test.clicked.connect(self._v15_test)
        grid.addWidget(train,1,0); grid.addWidget(test,1,1)
        self.c_train_status = QLabel("V0.15 visual training budget = 0%."); self.c_train_status.setWordWrap(True); grid.addWidget(self.c_train_status,2,0,1,2)

        examples = QGroupBox("Teaching examples"); eg = QVBoxLayout(examples); rl.addWidget(examples)
        ex = QPlainTextEdit(); ex.setReadOnly(True); ex.setMaximumHeight(170)
        ex.setPlainText(
            "hello\n"
            "what is acceleration?\n"
            "what does it depend on?\n"
            "why?\n"
            "fluxion means acceleration\n"
            "what is fluxion?\n"
            "speed is distance divided by time\n"
            "what is speed?\n"
            "remember that orbix is a sensor\n"
            "what is orbix?\n"
            "what did I just teach you?\n"
            "compare speed and acceleration"
        )
        eg.addWidget(ex)

        rl.addWidget(QLabel("Conversation / language memory audit"))
        self.c_audit = QPlainTextEdit(); self.c_audit.setReadOnly(True); rl.addWidget(self.c_audit, 1)
        save = QPushButton("Save V0.15 Checkpoint"); save.clicked.connect(self._save_v15); rl.addWidget(save)
        return root

    def _v15_send(self):
        text = self.c_input.text().strip()
        if not text: return
        try:
            reply = self.cognitive.talk(text)
            self.c_chat.appendPlainText(f"YOU:  {text}\nAPCN: {reply.text}\n")
            self.c_status.setText(
                f"act={reply.act} • confidence={reply.confidence:.2f} • learned={reply.learned}"
                + (f" • concept={reply.concept}" if reply.concept else "")
            )
            self.c_input.clear(); self._save_v15(silent=True); self._refresh_v15()
        except Exception as exc:
            QMessageBox.critical(self, "Conversation failed", str(exc))

    def _v15_train(self):
        try:
            row = self.cognitive.language_only_train(self.c_steps.value())
            self.c_train_status.setText(
                f"language +{row['experiences_added']} • visual +{row['visual_experiences_added']} • correct-before-learning {row['correct_before_learning_rate']:.1%}"
            )
            self._save_v15(silent=True); self._refresh_v15(last=row)
        except Exception as exc:
            QMessageBox.critical(self, "Language training failed", str(exc))

    def _v15_test(self):
        try:
            samples = max(100, min(1000, self.c_steps.value()))
            row = self.cognitive.test_rich_language(samples)
            self.c_train_status.setText(
                f"HELD-OUT • exact {row['exact']:.1%} • intent {row['intent']:.1%} • relation {row['relation']:.1%} • operator {row['operator']:.1%} • frozen={row['memory_frozen']}"
            )
            self._save_v15(silent=True); self._refresh_v15(last=row)
        except Exception as exc:
            QMessageBox.critical(self, "Language test failed", str(exc))

    def _refresh_v15(self, last=None):
        if not hasattr(self, "c_audit"): return
        audit = self.cognitive.conversation_memory_audit()
        if last is not None:
            audit = {"last_operation": last, **audit}
        self.c_audit.setPlainText(json.dumps(audit, indent=2, default=str))

    def _save_v15(self, silent: bool = False):
        try:
            self.cognitive.save("outputs/v0_15")
            if not silent and hasattr(self, "c_status"):
                self.c_status.setText("V0.15 checkpoint saved.")
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "Save failed", str(exc))

    def closeEvent(self, event):
        try:
            if hasattr(self, "_camera_stop"):
                self._camera_stop(silent=True)
            self.cognitive.save("outputs/v0_15")
        except Exception:
            pass
        event.accept()


def run_app(seed: int = 15) -> int:
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    win = APCNV15Window(seed); win.show(); return app.exec()
