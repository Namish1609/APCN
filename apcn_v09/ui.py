from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFrame, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSpinBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QPlainTextEdit,
)

from apcn_v08.session import TrainingSessionV08, StepResult
from apcn_v08.testing_v082 import BulkTestReport, run_bulk_test
from apcn_v08.ui_simple import APP_STYLE, SceneCanvas, FiringGraph
from .semantic import semantic_equal
from .session import LanguageStep, SemanticSessionV09
from .testing import SemanticTestReport
from .teacher import LanguageEpisode


class HistoryGraph(QWidget):
    """Read-only test-history graph drawn directly with Qt."""
    def __init__(self):
        super().__init__()
        self.history: List[Dict[str, float]] = []
        self.labels: List[str] = []
        self.setMinimumHeight(190)

    def set_history(self, history: List[Dict[str, float]], labels: List[str]) -> None:
        self.history = list(history)
        self.labels = list(labels)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#0b121c"))
        p.setPen(QColor("#e8f1fd")); p.setFont(QFont("Sans Serif", 10, 700))
        p.drawText(10, 19, "TEST HISTORY — accuracy vs learned episodes")
        box = QRectF(48, 34, max(60, self.width()-64), max(70, self.height()-56))
        p.setPen(QPen(QColor("#33465e"), 1)); p.drawRect(box)
        for frac in (0, .25, .5, .75, 1):
            y = box.bottom() - box.height()*frac
            p.setPen(QPen(QColor("#243448"), 1)); p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))
            p.setPen(QColor("#8296ad")); p.setFont(QFont("Sans Serif", 7)); p.drawText(5, int(y+3), f"{int(frac*100)}%")
        if not self.history:
            p.setPen(QColor("#8296ad")); p.drawText(box, Qt.AlignmentFlag.AlignCenter, "Run a test to start the graph")
            return
        max_ep = max(1.0, max(float(h.get("episodes", 0)) for h in self.history))
        colors = [QColor("#56c7ff"), QColor("#69e39c"), QColor("#ffc857"), QColor("#c49cff")]
        for li, label in enumerate(self.labels):
            pts = []
            for h in self.history:
                x = box.left() + box.width() * float(h.get("episodes", 0)) / max_ep
                y = box.bottom() - box.height() * float(np.clip(h.get(label, 0.0), 0, 1))
                pts.append(QPointF(x, y))
            p.setPen(QPen(colors[li % len(colors)], 2))
            for a, b in zip(pts, pts[1:]): p.drawLine(a, b)
            for q in pts: p.drawEllipse(q, 2.5, 2.5)
            y = box.top() + 10 + li*14
            p.drawLine(QPointF(box.left()+5, y), QPointF(box.left()+20, y))
            p.setPen(QColor("#dce7f5")); p.drawText(int(box.left()+24), int(y+3), label)


class APCNV09Window(QMainWindow):
    def __init__(self, seed: int = 9):
        super().__init__()
        self.setWindowTitle("APCN V0.9 — Grounded Semantic Language Studio")
        self.resize(1366, 768)
        self.setMinimumSize(1100, 650)
        self.visual = TrainingSessionV08(seed=seed)
        self.language = SemanticSessionV09(seed=seed)
        self.visual_current: Optional[StepResult] = None
        self.language_current: Optional[LanguageStep] = None
        self.language_context: Optional[LanguageEpisode] = None
        self.visual_history: List[Dict[str, float]] = []
        self.semantic_history: List[Dict[str, float]] = []
        self._build_ui()
        self._new_visual(); self._new_language(); self._refresh_concepts()

    def _card(self):
        f = QFrame(); f.setObjectName("card"); return f

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        outer = QVBoxLayout(root); outer.setContentsMargins(8, 6, 8, 6); outer.setSpacing(6)
        head = QHBoxLayout(); title = QLabel("APCN V0.9  •  GROUNDED SEMANTIC LANGUAGE STUDIO"); title.setFont(QFont("Sans Serif", 13, 800)); head.addWidget(title); head.addStretch(); self.counter = QLabel("visual 0 • language 0"); head.addWidget(self.counter); outer.addLayout(head)
        self.tabs = QTabWidget(); self.tabs.addTab(self._visual_tab(), "Perception"); self.tabs.addTab(self._language_tab(), "Language + Context"); self.tabs.addTab(self._testing_tab(), "Testing Ground"); self.tabs.addTab(self._concept_tab(), "Concept Memory"); outer.addWidget(self.tabs, 1)

    def _visual_tab(self):
        page = QWidget(); row = QHBoxLayout(page); row.setContentsMargins(6, 6, 6, 6)
        self.canvas = SceneCanvas(False); row.addWidget(self.canvas, 3)
        side = self._card(); side.setMaximumWidth(470); s = QVBoxLayout(side)
        h = QLabel("Perceptual grounding retained from V0.8.2"); h.setFont(QFont("Sans Serif", 11, 800)); s.addWidget(h)
        picks = QHBoxLayout(); self.vcolor = QComboBox(); self.vcolor.addItem("random"); self.vcolor.addItems(self.visual.teacher.color_words); self.vshape = QComboBox(); self.vshape.addItem("random"); self.vshape.addItems(self.visual.teacher.shape_words); picks.addWidget(self.vcolor); picks.addWidget(self.vshape); s.addLayout(picks)
        b = QPushButton("New / Test Example — no learning"); b.setObjectName("primary"); b.clicked.connect(self._new_visual); s.addWidget(b)
        b = QPushButton("Teach Visible Example"); b.setObjectName("learnButton"); b.clicked.connect(self._teach_visual); s.addWidget(b)
        batch = QHBoxLayout(); self.vbatch = QSpinBox(); self.vbatch.setRange(10, 5000); self.vbatch.setValue(500); b = QPushButton("Auto Train Batch"); b.clicked.connect(self._train_visual); batch.addWidget(self.vbatch); batch.addWidget(b); s.addLayout(batch)
        self.vresult = QLabel("—"); self.vresult.setWordWrap(True); s.addWidget(self.vresult)
        self.firing = FiringGraph(); s.addWidget(self.firing, 1); row.addWidget(side, 2); return page

    def _language_tab(self):
        page = QWidget(); row = QHBoxLayout(page); row.setContentsMargins(7, 7, 7, 7); row.setSpacing(8)
        left = self._card(); ll = QVBoxLayout(left)
        h = QLabel("Grounded semantic lesson"); h.setFont(QFont("Sans Serif", 12, 800)); ll.addWidget(h)
        top = QHBoxLayout(); self.lang_mode = QComboBox(); self.lang_mode.addItems(["simple relation", "group / and", "sequence / then", "negation / not", "context / pronoun"]); b = QPushButton("New sentence — no learning"); b.setObjectName("primary"); b.clicked.connect(self._new_language); top.addWidget(self.lang_mode); top.addWidget(b); ll.addLayout(top)
        self.context_label = QLabel("Context: none"); self.context_label.setWordWrap(True); self.context_label.setStyleSheet("color:#9db0c7"); ll.addWidget(self.context_label)
        self.utterance = QLabel("Utterance: —"); self.utterance.setWordWrap(True); self.utterance.setFont(QFont("Sans Serif", 12, 700)); ll.addWidget(self.utterance)
        meanings = QHBoxLayout(); self.teacher_sem = QPlainTextEdit(); self.teacher_sem.setReadOnly(True); self.apcn_sem = QPlainTextEdit(); self.apcn_sem.setReadOnly(True); meanings.addWidget(self._text_card("Teacher/world meaning", self.teacher_sem), 1); meanings.addWidget(self._text_card("APCN interpretation", self.apcn_sem), 1); ll.addLayout(meanings, 1)
        self.lang_check = QLabel("—"); ll.addWidget(self.lang_check)
        b = QPushButton("Teach This Semantic Experience"); b.setObjectName("learnButton"); b.clicked.connect(self._teach_language); ll.addWidget(b); row.addWidget(left, 3)
        right = self._card(); rl = QVBoxLayout(right)
        h = QLabel("Automatic language curriculum"); h.setFont(QFont("Sans Serif", 11, 800)); rl.addWidget(h)
        info = QLabel("Stages: lexical relations → intent → composition → and/then/not → contextual reference. English cue mappings are learned; internal operators remain generic primitives."); info.setWordWrap(True); info.setStyleSheet("color:#9db0c7"); rl.addWidget(info)
        self.lbatch = QSpinBox(); self.lbatch.setRange(50, 10000); self.lbatch.setValue(800); rl.addWidget(self.lbatch)
        b = QPushButton("Auto Train Language Batch"); b.clicked.connect(self._train_language); rl.addWidget(b)
        self.phase_label = QLabel("phase: —"); self.phase_label.setWordWrap(True); rl.addWidget(self.phase_label)
        self.cue_text = QPlainTextEdit(); self.cue_text.setReadOnly(True); rl.addWidget(self.cue_text, 1); row.addWidget(right, 2); return page

    def _testing_tab(self):
        page = QWidget(); outer = QVBoxLayout(page); outer.setContentsMargins(7, 7, 7, 7); outer.setSpacing(7)
        controls = self._card(); c = QHBoxLayout(controls); self.test_mode = QComboBox(); self.test_mode.addItems(["Perception", "Language semantics"]); self.test_n = QSpinBox(); self.test_n.setRange(50, 5000); self.test_n.setValue(500); self.test_hard = QComboBox(); self.test_hard.addItem("Normal language templates", False); self.test_hard.addItem("Held-out language templates", True); run = QPushButton("Run Test — memory frozen"); run.setObjectName("primary"); run.clicked.connect(self._run_test); c.addWidget(self.test_mode); c.addWidget(QLabel("samples")); c.addWidget(self.test_n); c.addWidget(self.test_hard); c.addWidget(run); c.addStretch(); self.memory_check = QLabel("memory unchanged: —"); c.addWidget(self.memory_check); outer.addWidget(controls)
        # V0.9 deliberately removes the redundant score cards.
        mats = QHBoxLayout(); self.matrix_a = QTableWidget(); self.matrix_b = QTableWidget(); self.matrix_a_title = QLabel("Matrix A"); self.matrix_b_title = QLabel("Matrix B"); mats.addWidget(self._matrix_card(self.matrix_a_title, self.matrix_a), 1); mats.addWidget(self._matrix_card(self.matrix_b_title, self.matrix_b), 1); outer.addLayout(mats, 3)
        # Requested split: failures and graphs.
        bottom = QHBoxLayout(); fail = self._card(); fl = QVBoxLayout(fail); fl.addWidget(QLabel("FAILURES — inspect what APCN got wrong")); self.failures = QPlainTextEdit(); self.failures.setReadOnly(True); fl.addWidget(self.failures); graph_card = self._card(); gl = QVBoxLayout(graph_card); self.graph = HistoryGraph(); gl.addWidget(self.graph); bottom.addWidget(fail, 1); bottom.addWidget(graph_card, 1); outer.addLayout(bottom, 2); return page

    def _concept_tab(self):
        page = QWidget(); row = QHBoxLayout(page); self.visual_concepts = QPlainTextEdit(); self.visual_concepts.setReadOnly(True); self.language_concepts = QPlainTextEdit(); self.language_concepts.setReadOnly(True); row.addWidget(self._text_card("Perceptual words", self.visual_concepts), 1); row.addWidget(self._text_card("Semantic cues / constructions", self.language_concepts), 1); return page

    def _text_card(self, title, widget):
        card = self._card(); l = QVBoxLayout(card); lab = QLabel(title); lab.setFont(QFont("Sans Serif", 9, 700)); l.addWidget(lab); l.addWidget(widget); return card

    def _matrix_card(self, label, table):
        card = self._card(); l = QVBoxLayout(card); label.setFont(QFont("Sans Serif", 9, 700)); l.addWidget(label); l.addWidget(table); return card

    def _choice(self, combo): return None if combo.currentText() == "random" else combo.currentText()
    def _update_counter(self): self.counter.setText(f"visual {self.visual.learner.episode_count} • language {self.language.learner.episode_count}")

    def _new_visual(self):
        r = self.visual.generate_preview(self._choice(self.vcolor), self._choice(self.vshape), difficulty=0.0, add_distractors=False); self.visual_current = r; ep = r.episode; tc, ts = ep.teacher_metadata['color'], ep.teacher_metadata['shape']; pc, ps = r.predicted_color or '?', r.predicted_shape or '?'; self.canvas.set_scene(ep.image, ep.attention_mask, f"TEST / NO LEARNING: {tc} {ts}", "dashed outline = teacher focus"); self.vresult.setText(f"Teacher: {ep.utterance}\nTruth: {tc} {ts}\nAPCN: {pc} {ps}\n{'✓ correct' if pc == tc and ps == ts else 'prediction differs — memory is frozen'}"); self.firing.set_trace(self.visual.learner.activation_trace(r.features, ep.utterance)); self._update_counter()

    def _teach_visual(self):
        if self.visual_current is None: return
        ep = self.visual_current.episode; self.visual.teach_current(ep.utterance, ep.image, ep.attention_mask); self.vresult.setText(self.vresult.text() + "\nLEARNED: this visible example was added once."); self._update_counter(); self._refresh_concepts()

    def _train_visual(self):
        n = self.vbatch.value(); last = None
        for i in range(n):
            last = self.visual.step()
            if i % 50 == 0: QApplication.processEvents()
        if last is not None:
            self.visual_current = last; ep = last.episode; self.canvas.set_scene(ep.image, ep.attention_mask, f"TRAINING SAMPLE — learned {n}", "actual generated training sample"); self.firing.set_trace(self.visual.learner.activation_trace(last.features, ep.utterance))
        self._update_counter(); self._refresh_concepts()

    def _make_language_step(self):
        mode = self.lang_mode.currentIndex(); self.language_context = None
        if mode == 1: ep = self.language.teacher.group()
        elif mode == 2: ep = self.language.teacher.sequence()
        elif mode == 3: ep = self.language.teacher.negated()
        elif mode == 4:
            first, ep = self.language.teacher.reference_pair(); self.language_context = first
        else: ep = self.language.teacher.simple()
        return LanguageStep(ep, self.language.learner.parse(ep.utterance, ep.discourse_focus), False)

    def _new_language(self):
        step = self._make_language_step(); self.language_current = step; ep = step.episode; self.context_label.setText("Context: " + (self.language_context.utterance if self.language_context else "none")); self.utterance.setText("Utterance:  " + ep.utterance); self.teacher_sem.setPlainText(ep.program.pretty()); self.apcn_sem.setPlainText("<parse failed>" if step.prediction is None else step.prediction.pretty()); ok = semantic_equal(step.prediction, ep.program); self.lang_check.setText("✓ semantic program matches" if ok else "✗ semantic program differs — preview is read-only"); self.lang_check.setStyleSheet("color:#69e39c" if ok else "color:#ffc857"); self.phase_label.setText(f"preview • learned language experiences: {self.language.learner.episode_count}"); self._refresh_cues(); self._update_counter()

    def _teach_language(self):
        if self.language_current is None: return
        if self.language_context is not None: self.language.learner.observe(self.language_context)
        self.language.learner.observe(self.language_current.episode); self.language.curriculum_index = self.language.learner.episode_count; self.lang_check.setText("LEARNED: utterance + grounded semantic demonstration added."); self._update_counter(); self._refresh_cues(); self._refresh_concepts()

    def _train_language(self):
        n = self.lbatch.value(); last = None
        for i in range(n):
            last = self.language.step()
            if i % 100 == 0: QApplication.processEvents()
        self.phase_label.setText(f"learned {n} curriculum steps • last phase={last.episode.phase if last else '?'}"); self._update_counter(); self._refresh_cues(); self._refresh_concepts(); self._new_language()

    def _refresh_cues(self):
        feats = ["relation:R0", "relation:R1", "relation:R2", "intent:QUERY", "intent:GOAL", "operator:GROUP", "operator:SEQUENCE", "operator:NEGATE", "reference:FOCUS"]
        lines = []
        for f in feats:
            cues = self.language.learner.top_cues(f, 3); lines.append(f + "\n  " + (", ".join(f"{x.cue} ({x.score:.2f})" for x in cues) if cues else "—"))
        self.cue_text.setPlainText("\n\n".join(lines))

    def _run_test(self):
        n = self.test_n.value(); semantic = self.test_mode.currentIndex() == 1; self.memory_check.setText("running…"); QApplication.processEvents()
        try:
            if semantic:
                self._render_semantic(self.language.test(n, held_out_templates=bool(self.test_hard.currentData())))
            else:
                self._render_visual(run_bulk_test(self.visual.learner, n, .82, seed=29090+n+len(self.visual_history)))
        except Exception as exc:
            QMessageBox.critical(self, "Test failed", str(exc)); self.memory_check.setText("failed")

    def _render_visual(self, rep: BulkTestReport):
        self.matrix_a_title.setText("COLOR CONFUSION — rows truth, columns prediction"); self.matrix_b_title.setText("SHAPE CONFUSION — rows truth, columns prediction"); self._fill_matrix(self.matrix_a, rep.color_labels, rep.color_confusion); self._fill_matrix(self.matrix_b, rep.shape_labels, rep.shape_confusion); self.memory_check.setText(f"memory unchanged: {rep.learner_episode_count_before} → {rep.learner_episode_count_after}"); self.failures.setPlainText("\n".join(f"truth {f.truth_color} {f.truth_shape} → pred {f.pred_color} {f.pred_shape}" for f in rep.failures) or "No failures in retained sample."); self.visual_history.append({"episodes": float(self.visual.learner.episode_count), "color": rep.color_accuracy, "shape": rep.shape_accuracy, "joint": rep.joint_accuracy}); self.graph.set_history(self.visual_history, ["color", "shape", "joint"])

    def _render_semantic(self, rep: SemanticTestReport):
        self.matrix_a_title.setText("INTENT CONFUSION — ASSERT / QUERY / GOAL"); self.matrix_b_title.setText("RELATION CONFUSION — learned internal relation IDs"); self._fill_matrix(self.matrix_a, rep.confusion_labels, rep.intent_confusion); self._fill_matrix(self.matrix_b, rep.relation_labels, rep.relation_confusion); self.memory_check.setText(f"memory unchanged: {rep.learner_episode_count_before} → {rep.learner_episode_count_after}"); chunks = [f"[{f.kind}] {f.utterance}\nEXPECTED\n{f.expected}\nPREDICTED\n{f.predicted}" for f in rep.failures]; self.failures.setPlainText("\n\n".join(chunks) or "No failures in retained sample."); self.semantic_history.append({"episodes": float(self.language.learner.episode_count), "exact": rep.exact_accuracy, "intent": rep.intent_accuracy, "relation": rep.relation_accuracy, "operator": rep.operator_accuracy}); self.graph.set_history(self.semantic_history, ["exact", "intent", "relation", "operator"])

    def _fill_matrix(self, table: QTableWidget, labels, mat):
        table.clear(); table.setRowCount(len(labels)); table.setColumnCount(len(labels)); table.setVerticalHeaderLabels(labels); table.setHorizontalHeaderLabels(labels)
        for r, row in enumerate(mat):
            for c, v in enumerate(row):
                item = QTableWidgetItem(str(v)); item.setTextAlignment(Qt.AlignmentFlag.AlignCenter); table.setItem(r, c, item)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def _refresh_concepts(self):
        rows = [f"{t:<12} quality={self.visual.learner.concept_quality(t):.3f} support={s.count}" for t, s in sorted(self.visual.learner.token_stats.items(), key=lambda kv: self.visual.learner.concept_quality(kv[0]), reverse=True)]
        self.visual_concepts.setPlainText("\n".join(rows[:100]) or "No perceptual concepts yet.")
        lines = []
        for f in sorted(self.language.learner.feature_totals):
            cues = self.language.learner.top_cues(f, 5); lines.append(f + "\n  " + ", ".join(f"{c.cue}:{c.score:.2f}" for c in cues))
        self.language_concepts.setPlainText("\n\n".join(lines) or "No semantic language cues yet.")


def run_app(seed: int = 9) -> int:
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion"); app.setStyleSheet(APP_STYLE)
    win = APCNV09Window(seed=seed); win.show()
    return app.exec()
