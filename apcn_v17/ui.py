from __future__ import annotations

from pathlib import Path
import json

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QPlainTextEdit, QGroupBox, QMessageBox, QSpinBox,
)

from apcn_v16.ui import APCNV16Window
from .semantics import SemanticClause
from .session import CognitiveSessionV17


class APCNV17Window(APCNV16Window):
    """V0.17 studio for compositional discourse semantics and explicit rules."""

    def __init__(self, seed: int = 17):
        super().__init__(seed)
        self.seed = seed
        self.setWindowTitle("APCN V0.17 — Structured Discourse Semantics")

        out17 = Path("outputs/v0_17")
        out16 = Path("outputs/v0_16")
        if (out17 / "session_v0_17.json").exists():
            self.cognitive = CognitiveSessionV17.load_checkpoint(out17, seed=seed)
            note = "loaded V0.17 structured-semantics checkpoint"
        elif (out16 / "session_v0_16.json").exists():
            self.cognitive = CognitiveSessionV17.from_v16_checkpoint(out16, seed=seed)
            self.cognitive.save(out17)
            note = "imported V0.16 cognition and added structured semantic/discourse memory"
        else:
            self.cognitive = CognitiveSessionV17(seed)
            note = "new V0.17 structured-semantics memory"

        self.perception = self.cognitive.visual
        self.query_engine = self.cognitive.query
        self.visual_history = self.cognitive.visual_test_history
        self.consolidation_history = self.cognitive.consolidation_history

        self.p_hint.setText(
            "V0.17 extends the semantic language itself rather than adding phrase lists. It supports explicit negation, cause, "
            "conditionals, temporal order, quantifiers and bounded discourse reference. V0.16 concept/world memory stays authoritative; "
            "V0.17 stores higher-order statements as inspectable semantic records and the surface generator still has no truth-memory access."
        )
        self.p_current.setText(f"V0.17: {note}")
        self.tabs.addTab(self._build_structured_tab(), "Structured Semantics")
        self._refresh_v17()
        self._refresh_v16()
        self._refresh_v15()
        self._refresh_header()

    def _build_structured_tab(self) -> QWidget:
        root = QWidget(); outer = QVBoxLayout(root); outer.setContentsMargins(10, 10, 10, 10)
        title = QLabel("Nested Semantic Operators + Discourse Memory")
        title.setFont(QFont("Sans Serif", 14, 800)); outer.addWidget(title)
        hint = QLabel(
            "Use this tab to inspect the semantic compiler directly. The normal Conversation tab also uses V0.17 now. "
            "Explicit teaching examples: 'remember that if battery is empty then device stops', 'remember that lamp turns off because power fails', "
            "'remember that every cat is a creature', and then ask truth/why/what-follows questions."
        )
        hint.setWordWrap(True); outer.addWidget(hint)

        row = QHBoxLayout(); self.s_input = QLineEdit(); self.s_input.setPlaceholderText("Example: lamp turns off because power fails")
        parse = QPushButton("Parse Structure"); parse.clicked.connect(self._v17_parse)
        teach = QPushButton("Store as Explicit Semantic Memory"); teach.clicked.connect(self._v17_teach_surface)
        row.addWidget(self.s_input, 1); row.addWidget(parse); row.addWidget(teach); outer.addLayout(row)

        self.s_output = QPlainTextEdit(); self.s_output.setReadOnly(True); outer.addWidget(self.s_output, 1)

        box = QGroupBox("Generate from explicit nested semantics"); bl = QVBoxLayout(box)
        self.s_semantic = QPlainTextEdit(); self.s_semantic.setMaximumHeight(190)
        self.s_semantic.setPlainText(json.dumps({
            "op": "IF",
            "slots": {},
            "tense": "atemporal",
            "children": [
                {"op": "PROPERTY", "slots": {"subject": "battery", "property": "empty"}, "tense": "present", "children": []},
                {"op": "EVENT", "slots": {"subject": "device", "event": "stop"}, "tense": "present", "children": []},
            ],
        }, indent=2))
        bl.addWidget(self.s_semantic)
        rr = QHBoxLayout(); self.s_limit = QSpinBox(); self.s_limit.setRange(1, 20); self.s_limit.setValue(10)
        gen = QPushButton("Generate + Semantic Roundtrip"); gen.clicked.connect(self._v17_generate)
        rr.addWidget(QLabel("Variants")); rr.addWidget(self.s_limit); rr.addWidget(gen); rr.addStretch(1); bl.addLayout(rr)
        outer.addWidget(box)

        examples = QGroupBox("Conversation examples"); el = QVBoxLayout(examples)
        ex = QPlainTextEdit(); ex.setReadOnly(True); ex.setMaximumHeight(155)
        ex.setPlainText(
            "remember that milo is a cat\n"
            "remember that it is active\n"
            "is it true that it is active?\n"
            "remember that every cat is a creature\n"
            "is it true that milo is a creature?\n"
            "remember that if battery is empty then device stops\n"
            "what follows if battery is empty?\n"
            "remember that lamp turns off because power fails\n"
            "what causes lamp turns off?\n"
            "remember that door opens before light turns on\n"
            "what happens before light turns on?\n"
            "remember that not sensor is active\n"
            "is it true that sensor is active?"
        )
        el.addWidget(ex); outer.addWidget(examples)

        outer.addWidget(QLabel("V0.17 semantic/discourse audit"))
        self.s_audit = QPlainTextEdit(); self.s_audit.setReadOnly(True); self.s_audit.setMaximumHeight(260); outer.addWidget(self.s_audit)
        save = QPushButton("Save V0.17 Checkpoint"); save.clicked.connect(lambda: self._save_v15(False)); outer.addWidget(save)
        return root

    def _v17_parse(self):
        text = self.s_input.text().strip()
        if not text:
            return
        try:
            row = self.cognitive.parse_structured(text)
            self.s_output.setPlainText(json.dumps(row, indent=2, default=str))
            self._refresh_v17(last=row)
        except Exception as exc:
            QMessageBox.critical(self, "V0.17 parse failed", str(exc))

    def _v17_teach_surface(self):
        text = self.s_input.text().strip()
        if not text:
            return
        try:
            parsed = self.cognitive.parse_structured(text)
            if not parsed.get("semantic"):
                raise ValueError("statement could not be compiled")
            clause = SemanticClause.from_dict(parsed["semantic"])
            row = self.cognitive.teach_structured_clause(clause)
            self.s_output.setPlainText(json.dumps(row, indent=2, default=str))
            self._save_v15(True); self._refresh_v17(last=row)
        except Exception as exc:
            QMessageBox.critical(self, "V0.17 teaching failed", str(exc))

    def _v17_generate(self):
        try:
            clause = SemanticClause.from_dict(json.loads(self.s_semantic.toPlainText()))
            row = self.cognitive.generate_structured(clause, self.s_limit.value())
            self.s_output.setPlainText(json.dumps(row, indent=2, default=str))
            self._refresh_v17(last=row)
        except Exception as exc:
            QMessageBox.critical(self, "V0.17 generation failed", str(exc))

    def _refresh_v17(self, last=None):
        if not hasattr(self, "s_audit"):
            return
        audit = self.cognitive.v17_memory_audit()
        if last is not None:
            audit = {"last_operation": last, **audit}
        self.s_audit.setPlainText(json.dumps(audit, indent=2, default=str))

    # V0.15/V0.16 inherited widgets dynamically call this method. Keep all saves
    # inside the V0.17 checkpoint tree.
    def _save_v15(self, silent: bool = False):
        try:
            self.cognitive.save("outputs/v0_17")
            if not silent and hasattr(self, "c_status"):
                self.c_status.setText("V0.17 checkpoint saved.")
            self._refresh_v17()
            self._refresh_v16()
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "Save failed", str(exc))

    def closeEvent(self, event):
        try:
            self.cognitive.save("outputs/v0_17")
        except Exception:
            pass
        event.accept()


def run_app(seed: int = 17) -> int:
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    win = APCNV17Window(seed); win.show(); return app.exec()
