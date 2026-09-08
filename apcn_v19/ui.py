from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .session import CognitiveSessionV19


class APCNV19Window(QMainWindow):
    """Focused V0.19 language + arithmetic test application."""

    def __init__(self, seed: int = 19):
        super().__init__()
        self.seed = seed
        self.output_dir = Path("outputs/v0_19")
        self.loaded_checkpoint = False
        self.cognitive = self._load_or_create()

        self.setWindowTitle("APCN V0.19 — Online English + Arithmetic")
        self.resize(980, 720)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)

        title = QLabel("APCN V0.19 — Language + Arithmetic Before World Model")
        title.setFont(QFont("Sans Serif", 15, 800))
        outer.addWidget(title)

        self.memory_state = QLabel()
        self.memory_state.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self.memory_state)

        hint = QLabel(
            "Primary gate: learn and generalize addition/subtraction language online without backpropagation. "
            "Examples: 'what is 37 plus 58?', 'subtract 7 from 30', or teach a new word with "
            "'remember that 2 dax 3 equals 5' and then ask 'what is 11 dax 8?'."
        )
        hint.setWordWrap(True)
        outer.addWidget(hint)

        self.chat = QPlainTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setFont(QFont("Consolas", 11))
        outer.addWidget(self.chat, 1)

        input_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Type an English or arithmetic request...")
        self.input.returnPressed.connect(self._send)
        send = QPushButton("Send")
        send.clicked.connect(self._send)
        input_row.addWidget(self.input, 1)
        input_row.addWidget(send)
        outer.addLayout(input_row)

        buttons = QHBoxLayout()
        clean = QPushButton("Start Clean Test Session")
        clean.clicked.connect(self._clean_session)
        save = QPushButton("Save Checkpoint")
        save.clicked.connect(self._save)
        audit = QPushButton("Show V0.19 Audit")
        audit.clicked.connect(self._show_audit)
        buttons.addWidget(clean)
        buttons.addWidget(save)
        buttons.addWidget(audit)
        buttons.addStretch(1)
        outer.addLayout(buttons)

        self._refresh_state()
        self.chat.appendPlainText("APCN: Ready. This window is focused on English construction learning and addition/subtraction.\n")

    def _load_or_create(self) -> CognitiveSessionV19:
        if (self.output_dir / "session_v0_19.json").exists():
            try:
                self.loaded_checkpoint = True
                return CognitiveSessionV19.load_checkpoint(self.output_dir, seed=self.seed)
            except Exception:
                pass
        self.loaded_checkpoint = False
        return CognitiveSessionV19(seed=self.seed, bootstrap_english=True)

    def _refresh_state(self) -> None:
        summary = self.cognitive.language_math_v19.summary()
        mode = "LOADED CHECKPOINT" if self.loaded_checkpoint else "CLEAN IN-MEMORY SESSION"
        self.memory_state.setText(
            f"Memory state: {mode} | learned demonstrations: {summary['demonstrations']} | "
            f"construction features: {summary['feature_count']} | backpropagation: OFF"
        )

    def _send(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self.chat.appendPlainText(f"YOU:  {text}")
        try:
            reply = self.cognitive.talk(text)
            self.chat.appendPlainText(f"APCN: {reply.text}\n")
        except Exception as exc:
            self.chat.appendPlainText(f"APCN ERROR: {exc}\n")
        self._refresh_state()

    def _clean_session(self) -> None:
        self.cognitive = CognitiveSessionV19(seed=self.seed, bootstrap_english=True)
        self.loaded_checkpoint = False
        self.chat.clear()
        self.chat.appendPlainText("APCN: Started a clean in-memory V0.19 test session. The saved checkpoint was not deleted.\n")
        self._refresh_state()

    def _save(self) -> None:
        try:
            self.cognitive.save(self.output_dir)
            self.loaded_checkpoint = True
            self._refresh_state()
            QMessageBox.information(self, "Saved", "V0.19 checkpoint saved to outputs/v0_19.")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _show_audit(self) -> None:
        import json

        QMessageBox.information(
            self,
            "V0.19 Architecture Audit",
            json.dumps(self.cognitive.v19_memory_audit(), indent=2),
        )

    def closeEvent(self, event):
        try:
            self.cognitive.save(self.output_dir)
        except Exception:
            pass
        event.accept()


def run_app(seed: int = 19) -> int:
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    win = APCNV19Window(seed=seed)
    win.show()
    return app.exec()
