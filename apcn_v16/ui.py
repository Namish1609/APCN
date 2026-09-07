from __future__ import annotations

from pathlib import Path
import json

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QPlainTextEdit, QGroupBox, QMessageBox, QSpinBox,
)

from apcn_v15.ui import APCNV15Window
from .bidirectional import SemanticFrame
from .session import CognitiveSessionV16


class APCNV16Window(APCNV15Window):
    """V0.16 language studio: the same learned constructions parse and realize."""

    def __init__(self, seed: int = 16):
        super().__init__(seed)
        self.seed = seed
        self.setWindowTitle("APCN V0.16 — Bidirectional Semantic Language Studio")

        out16 = Path("outputs/v0_16")
        out15 = Path("outputs/v0_15")
        if (out16 / "session_v0_16.json").exists():
            self.cognitive = CognitiveSessionV16.load_checkpoint(out16, seed=seed)
            note = "loaded V0.16 bidirectional-language checkpoint"
        elif (out15 / "session_v0_15.json").exists():
            self.cognitive = CognitiveSessionV16.from_v15_checkpoint(out15, seed=seed)
            self.cognitive.save(out16)
            note = "imported V0.15 cognition and added bidirectional construction memory"
        else:
            self.cognitive = CognitiveSessionV16(seed)
            note = "new V0.16 bidirectional-language memory"

        self.perception = self.cognitive.visual
        self.query_engine = self.cognitive.query
        self.visual_history = self.cognitive.visual_test_history
        self.consolidation_history = self.cognitive.consolidation_history

        self.p_hint.setText(
            "V0.16 keeps concept/world memory authoritative and makes language bidirectional: the same learned construction memory "
            "maps English into explicit semantic frames and maps semantic answer frames back into varied English. The surface generator "
            "has no reference to world memory and therefore cannot independently decide facts."
        )
        self.p_current.setText(f"V0.16: {note}")
        self.tabs.addTab(self._build_bidirectional_tab(), "Bidirectional Language")
        self._refresh_v16()
        self._refresh_v15()
        self._refresh_header()

    def _build_bidirectional_tab(self) -> QWidget:
        root = QWidget(); outer = QVBoxLayout(root); outer.setContentsMargins(10,10,10,10)
        title = QLabel("English ↔ Semantic Frame — Shared Construction Memory")
        title.setFont(QFont("Sans Serif", 14, 800)); outer.addWidget(title)
        hint = QLabel(
            "Parse English into a semantic frame, generate paraphrases from that frame, and verify parse(generate(S)) ≈ S. "
            "Generation is surface realization only: concept/world truth is produced upstream by the semantic responder."
        )
        hint.setWordWrap(True); outer.addWidget(hint)

        row = QHBoxLayout(); self.b_input = QLineEdit(); self.b_input.setPlaceholderText("Example: what is acceleration")
        parse = QPushButton("Parse + Paraphrase"); parse.clicked.connect(self._v16_parse)
        row.addWidget(self.b_input, 1); row.addWidget(parse); outer.addLayout(row)

        self.b_output = QPlainTextEdit(); self.b_output.setReadOnly(True); outer.addWidget(self.b_output, 1)

        box = QGroupBox("Generate directly from explicit semantics"); bl = QVBoxLayout(box)
        self.b_semantic = QPlainTextEdit(); self.b_semantic.setMaximumHeight(130)
        self.b_semantic.setPlainText(json.dumps({"op":"STATE_DEFINITION","slots":{"concept":"acceleration","definition":"velocity change divided by time"}}, indent=2))
        bl.addWidget(self.b_semantic)
        rr = QHBoxLayout(); self.b_limit = QSpinBox(); self.b_limit.setRange(1, 20); self.b_limit.setValue(10)
        gen = QPushButton("Generate + Roundtrip Test"); gen.clicked.connect(self._v16_generate)
        rr.addWidget(QLabel("Variants")); rr.addWidget(self.b_limit); rr.addWidget(gen); rr.addStretch(1); bl.addLayout(rr)
        outer.addWidget(box)

        self.b_audit = QPlainTextEdit(); self.b_audit.setReadOnly(True); self.b_audit.setMaximumHeight(230); outer.addWidget(self.b_audit)
        save = QPushButton("Save V0.16 Checkpoint"); save.clicked.connect(lambda: self._save_v15(False)); outer.addWidget(save)
        return root

    def _v16_parse(self):
        text = self.b_input.text().strip()
        if not text: return
        try:
            row = self.cognitive.paraphrase(text, self.b_limit.value())
            self.b_output.setPlainText(json.dumps(row, indent=2, default=str))
            self._refresh_v16(last=row)
        except Exception as exc:
            QMessageBox.critical(self, "V0.16 parse failed", str(exc))

    def _v16_generate(self):
        try:
            frame = SemanticFrame.from_dict(json.loads(self.b_semantic.toPlainText()))
            row = self.cognitive.generate_from_semantics(frame, self.b_limit.value())
            self.b_output.setPlainText(json.dumps(row, indent=2, default=str))
            self._refresh_v16(last=row)
        except Exception as exc:
            QMessageBox.critical(self, "V0.16 generation failed", str(exc))

    def _refresh_v16(self, last=None):
        if not hasattr(self, "b_audit"): return
        audit = self.cognitive.v16_memory_audit()
        if last is not None:
            audit = {"last_operation": last, **audit}
        self.b_audit.setPlainText(json.dumps(audit, indent=2, default=str))

    # The inherited V0.15 conversation widgets call this dynamically. Redirect
    # every save into the V0.16 checkpoint so no V0.16 state is written beneath
    # outputs/v0_15.
    def _save_v15(self, silent: bool = False):
        try:
            self.cognitive.save("outputs/v0_16")
            if not silent and hasattr(self, "c_status"):
                self.c_status.setText("V0.16 checkpoint saved.")
            self._refresh_v16()
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "Save failed", str(exc))

    def closeEvent(self, event):
        try:
            self.cognitive.save("outputs/v0_16")
        except Exception:
            pass
        event.accept()


def run_app(seed: int = 16) -> int:
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    win = APCNV16Window(seed); win.show(); return app.exec()
