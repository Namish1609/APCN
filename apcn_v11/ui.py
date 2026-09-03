from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import json

from PyQt6.QtCore import Qt, QRectF, QPointF, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QBrush
from PyQt6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
    QProgressBar, QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QPlainTextEdit,
)

from apcn_v10.ui_v102 import APCNV102Window
from apcn_v10.ui_v101_widgets import SpikingGraph
from apcn_v10.ui_widgets import STYLE

from .session import CognitiveSessionV11


class ConsolidationComparison(QWidget):
    """Before/after view for the most recent automatic consolidation cycles."""

    def __init__(self):
        super().__init__()
        self.history: List[Dict[str, object]] = []
        self.setMinimumHeight(210)

    def set_history(self, rows: List[Dict[str, object]]) -> None:
        self.history = list(rows)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#0d151f"))
        p.setPen(QColor("#e9f2ff")); p.setFont(QFont("Sans Serif", 10, 700))
        p.drawText(12, 19, "Consolidation before → after")
        box = QRectF(48, 34, max(80, self.width()-64), max(70, self.height()-68))
        p.setPen(QPen(QColor("#40536b"), 1)); p.drawRect(box)
        for frac in (0.0, .25, .5, .75, 1.0):
            y = box.bottom() - frac * box.height()
            p.setPen(QPen(QColor("#263648"), 1)); p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))
            p.setPen(QColor("#8fa2b9")); p.drawText(8, int(y+4), str(int(frac*100)))
        if not self.history:
            p.setPen(QColor("#8fa2b9")); p.drawText(box, Qt.AlignmentFlag.AlignCenter,
                "Run an automatic consolidation cycle")
            return
        metrics = (
            ("visual_joint", QColor("#61e294"), "visual joint"),
            ("visual_shape", QColor("#ffc857"), "shape"),
            ("language_exact", QColor("#54d2ff"), "language exact"),
            ("language_intent", QColor("#c297ff"), "intent"),
        )
        # Each cycle occupies two x positions: before and after.
        total_slots = max(2, len(self.history) * 2)
        for key, color, _ in metrics:
            pts = []
            for ci, row in enumerate(self.history):
                for offset, phase in ((0, "before"), (1, "after")):
                    x = box.left() + ((ci*2 + offset) + .5) / total_slots * box.width()
                    val = float(row.get(phase, {}).get(key, 0.0))
                    y = box.bottom() - max(0.0, min(1.0, val)) * box.height()
                    pts.append(QPointF(x, y))
            p.setPen(QPen(color, 2))
            for a, b in zip(pts, pts[1:]): p.drawLine(a, b)
            p.setBrush(QBrush(color)); p.setPen(QPen(color, 1))
            for q in pts: p.drawEllipse(q, 3.4, 3.4)
        p.setFont(QFont("Sans Serif", 8))
        x = 54
        for _, color, label in metrics:
            p.setPen(color); p.drawText(x, self.height()-9, label); x += 96


class ConsolidationWorker(QThread):
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, session: CognitiveSessionV11, *, visual_test: int, language_test: int,
                 visual_train: int, language_train: int, difficulty: float):
        super().__init__()
        self.session = session
        self.visual_test = int(visual_test)
        self.language_test = int(language_test)
        self.visual_train = int(visual_train)
        self.language_train = int(language_train)
        self.difficulty = float(difficulty)

    def run(self):
        try:
            self.progress.emit(5, "diagnosing visual confusions")
            v0 = self.session.test_visual(self.visual_test, self.difficulty)
            self.progress.emit(18, "diagnosing language constructions and reference")
            l0 = self.session.test_language(self.language_test)
            before = {
                "visual_joint": v0.joint_accuracy,
                "visual_shape": v0.shape_accuracy,
                "language_exact": l0.exact_accuracy,
                "language_intent": l0.intent_accuracy,
                "language_reference": l0.skill_accuracy.get("reference", 0.0),
            }
            planned = [p.__dict__ for p in self.session.prescriptions(16)]
            self.progress.emit(35, "targeted visual minimal-pair learning")
            vt = self.session.consolidate_visual(self.visual_train)
            self.progress.emit(58, "targeted language construction learning")
            lt = self.session.consolidate_language(self.language_train)
            self.progress.emit(76, "retesting visual knowledge")
            v1 = self.session.test_visual(self.visual_test, self.difficulty)
            self.progress.emit(88, "retesting language and discourse identity")
            l1 = self.session.test_language(self.language_test)
            after = {
                "visual_joint": v1.joint_accuracy,
                "visual_shape": v1.shape_accuracy,
                "language_exact": l1.exact_accuracy,
                "language_intent": l1.intent_accuracy,
                "language_reference": l1.skill_accuracy.get("reference", 0.0),
            }
            graph = self.session.sync_graph()
            ambiguous = len(self.session.consolidation.visual_ambiguities(
                self.session.visual.learner, self.session.visual.teacher.shape_words, limit=30))
            objective = self.session.consolidation.objective(
                error_rate=1.0 - 0.5*(v1.joint_accuracy+l1.exact_accuracy),
                active_edges=int(graph.get("edges", 0)),
                ambiguous_pairs=ambiguous,
            )
            self.session.save("outputs/v0_11")
            self.progress.emit(100, "cycle complete")
            self.completed.emit({
                "before": before, "after": after,
                "visual_training": vt, "language_training": lt,
                "prescriptions": planned, "diagnostic_objective": objective,
            })
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class APCNV11Window(APCNV102Window):
    def __init__(self, seed: int = 11):
        super().__init__(seed)
        self.seed = seed
        self.setWindowTitle("APCN V0.11 — Self-Consolidating Learning Studio")

        # Prefer an existing V0.11 checkpoint. Otherwise start clean; V0.10
        # migration is explicit in the Consolidation tab.
        out = Path("outputs/v0_11")
        if (out / "visual_memory_v0_11.json").exists() or (out / "language_memory_v0_11.json").exists():
            self.cognitive = CognitiveSessionV11.from_memories(
                seed=seed,
                visual_memory=out / "visual_memory_v0_11.json",
                language_memory=out / "language_memory_v0_11.json",
                concept_memory=out / "concept_store_v0_11.json",
            )
        else:
            self.cognitive = CognitiveSessionV11(seed)
        self.perception = self.cognitive.visual
        self.query_engine = self.cognitive.query
        self.visual_history = self.cognitive.visual_test_history
        self.consolidation_history: List[Dict[str, object]] = []
        self.consolidation_worker: Optional[ConsolidationWorker] = None

        self.tabs.insertTab(3, self._consolidation_tab(), "Consolidation")
        self._refresh_competence()
        self._refresh_definition_table()
        self._refresh_header()
        self._refresh_consolidation()

    def _consolidation_tab(self) -> QWidget:
        page = QWidget(); outer = QVBoxLayout(page)
        outer.setContentsMargins(7, 7, 7, 7); outer.setSpacing(7)

        top = self._card(); tl = QHBoxLayout(top); tl.setContentsMargins(9, 7, 9, 7)
        self.c_visual_test = QSpinBox(); self.c_visual_test.setRange(100, 3000); self.c_visual_test.setValue(300)
        self.c_language_test = QSpinBox(); self.c_language_test.setRange(120, 3000); self.c_language_test.setValue(360)
        self.c_visual_train = QSpinBox(); self.c_visual_train.setRange(100, 10000); self.c_visual_train.setValue(400)
        self.c_language_train = QSpinBox(); self.c_language_train.setRange(100, 10000); self.c_language_train.setValue(500)
        self.c_run = QPushButton("Run 1 Automatic Consolidation Cycle")
        self.c_run.setObjectName("primary"); self.c_run.clicked.connect(self._start_consolidation)
        self.c_migrate = QPushButton("Migrate V0.10 Memory"); self.c_migrate.clicked.connect(self._migrate_v010)
        for label, widget in (("visual test", self.c_visual_test), ("language test", self.c_language_test),
                              ("visual teach", self.c_visual_train), ("language teach", self.c_language_train)):
            tl.addWidget(QLabel(label)); tl.addWidget(widget)
        tl.addWidget(self.c_run, 1); tl.addWidget(self.c_migrate)
        outer.addWidget(top)

        self.c_progress = QProgressBar(); self.c_progress.setRange(0, 100); self.c_progress.setValue(0)
        self.c_progress.setFormat("waiting")
        outer.addWidget(self.c_progress)

        main = QSplitter(Qt.Orientation.Horizontal)
        left = QSplitter(Qt.Orientation.Vertical)

        priority_card = self._card(); pl = QVBoxLayout(priority_card); pl.setContentsMargins(7,7,7,7)
        ph = QLabel("What APCN thinks it should learn next"); ph.setFont(QFont("Sans Serif", 10, 800)); pl.addWidget(ph)
        self.c_priorities = QTableWidget(); self.c_priorities.setColumnCount(5)
        self.c_priorities.setHorizontalHeaderLabels(["Domain", "Target", "Contrast", "Priority", "Reason"])
        self.c_priorities.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.c_priorities.verticalHeader().setVisible(False); pl.addWidget(self.c_priorities)
        left.addWidget(priority_card)

        compare_card = self._card(); cl = QVBoxLayout(compare_card); cl.setContentsMargins(7,7,7,7)
        self.c_curve = ConsolidationComparison(); cl.addWidget(self.c_curve)
        left.addWidget(compare_card); left.setSizes([310, 260])
        main.addWidget(left)

        right = QSplitter(Qt.Orientation.Vertical)
        graph_card = self._card(); gl = QVBoxLayout(graph_card); gl.setContentsMargins(7,7,7,7)
        gl.addWidget(QLabel("Unified concept graph — high-confidence active structure"))
        self.c_graph = SpikingGraph("UNIFIED CONCEPT GRAPH")
        gl.addWidget(self.c_graph); right.addWidget(graph_card)

        audit_card = self._card(); al = QVBoxLayout(audit_card); al.setContentsMargins(7,7,7,7)
        al.addWidget(QLabel("Memory + discourse audit"))
        self.c_audit = QPlainTextEdit(); self.c_audit.setReadOnly(True); al.addWidget(self.c_audit)
        right.addWidget(audit_card); right.setSizes([330, 240])
        main.addWidget(right); main.setSizes([790, 530])
        outer.addWidget(main, 1)
        return page

    def _graph_trace(self) -> Dict[str, object]:
        self.cognitive.sync_graph()
        graph = self.cognitive.graph
        rows = sorted(graph.nodes.values(), key=lambda n: (n.confidence, n.support), reverse=True)[:30]
        selected = {n.id for n in rows}
        nodes = []
        for n in rows:
            if n.kind == "lexical": kind = "word"
            elif n.kind == "discovered_family": kind = "family"
            elif n.kind.startswith("semantic_operator"): kind = "operator"
            elif n.kind.startswith("semantic_"): kind = "semantic"
            else: kind = "concept"
            firing = max(0.05, min(1.0, .65*n.confidence + .35*min(1.0, n.support/60.0)))
            nodes.append({"id": n.id, "label": n.label, "kind": kind, "firing": firing})
        edges = [
            {"src": e.src, "dst": e.dst, "weight": e.weight, "kind": e.relation}
            for e in graph.edges.values() if e.src in selected and e.dst in selected
        ]
        return {"nodes": nodes, "edges": edges}

    def _format_audit(self) -> str:
        a = self.cognitive.memory_audit()
        proto = a["visual_prototypes"]
        err = a["error_memory"]
        graph = a["concept_graph"]
        disc = a["discourse_working_memory"]
        cons = a["language_constructions"]
        return (
            f"EXPERIENCE VS RETAINED MEMORY\n"
            f"visual experiences seen: {a['visual_episodes_seen']}\n"
            f"raw visual examples retained: {a['visual_raw_examples_retained']}\n"
            f"visual token-stat records: {a['visual_token_stats']}\n"
            f"bounded visual prototypes: {proto['total_prototypes']} / max {proto['max_prototypes_per_token']} per token\n\n"
            f"language experiences seen: {a['language_episodes_seen']}\n"
            f"raw sentences retained: {a['language_raw_sentences_retained']}\n"
            f"unique learned cues/ngrams: {a['language_unique_cues']}\n"
            f"induced construction prefixes: {cons['prefix_constructions']}\n\n"
            f"error signatures: {err['signature_count']}\n"
            f"representative failure strings retained: {err['representative_examples_retained']}\n\n"
            f"unified graph: {graph['nodes']} nodes / {graph['edges']} edges\n"
            f"working discourse entities: {disc['entity_count']}\n"
            f"current discourse focus: {disc['focus']}\n\n"
            "Training episodes are evidence counts, not an archive of every image/sentence."
        )

    def _refresh_consolidation(self):
        if not hasattr(self, "c_priorities"):
            return
        rows = self.cognitive.prescriptions(16)
        self.c_priorities.setRowCount(len(rows))
        for r, p in enumerate(rows):
            vals = (p.domain, p.target, p.contrast, f"{p.priority:.3f}", p.reason)
            for c, val in enumerate(vals):
                item = QTableWidgetItem(str(val)); item.setTextAlignment(Qt.AlignmentFlag.AlignCenter if c < 4 else Qt.AlignmentFlag.AlignLeft)
                self.c_priorities.setItem(r, c, item)
        self.c_audit.setPlainText(self._format_audit())
        self.c_graph.set_trace(self._graph_trace())
        self.c_curve.set_history(self.consolidation_history)

    def _start_consolidation(self):
        if self.p_timer.isActive() or self.l_timer.isActive():
            QMessageBox.information(self, "Pause training first", "Pause Perception/Language training before consolidation so one memory writer is active at a time.")
            return
        if self.consolidation_worker is not None and self.consolidation_worker.isRunning():
            return
        self.c_run.setEnabled(False); self.c_migrate.setEnabled(False)
        self.c_progress.setValue(1); self.c_progress.setFormat("starting")
        worker = ConsolidationWorker(
            self.cognitive,
            visual_test=self.c_visual_test.value(), language_test=self.c_language_test.value(),
            visual_train=self.c_visual_train.value(), language_train=self.c_language_train.value(),
            difficulty=.82,
        )
        worker.progress.connect(self._consolidation_progress)
        worker.completed.connect(self._consolidation_done)
        worker.failed.connect(self._consolidation_failed)
        self.consolidation_worker = worker; worker.start()

    def _consolidation_progress(self, pct: int, text: str):
        self.c_progress.setValue(int(pct)); self.c_progress.setFormat(f"{pct}% • {text}")

    def _consolidation_done(self, result: object):
        row = dict(result)
        self.consolidation_history.append(row)
        self.c_progress.setValue(100); self.c_progress.setFormat("100% • cycle complete")
        self.c_run.setEnabled(True); self.c_migrate.setEnabled(True)
        self.visual_history = self.cognitive.visual_test_history
        self._refresh_header(); self._refresh_competence(); self._refresh_consolidation()
        if hasattr(self, "v_graph"): self.v_graph.set_history(self.visual_history)
        if hasattr(self, "lt_graph"): self.lt_graph.set_history(self.cognitive.language_test_history)

    def _consolidation_failed(self, text: str):
        self.c_run.setEnabled(True); self.c_migrate.setEnabled(True)
        self.c_progress.setFormat("failed")
        QMessageBox.critical(self, "Consolidation failed", text)

    def _migrate_v010(self):
        if self.p_timer.isActive() or self.l_timer.isActive():
            QMessageBox.information(self, "Pause training first", "Pause training before replacing the active memory.")
            return
        visual = Path("outputs/v0_10/perception/concept_memory_v0_8.json")
        language = Path("outputs/v0_10/language_memory_v0_10.json")
        concepts = Path("outputs/v0_10/concept_store_v0_10.json")
        found = [str(p) for p in (visual, language, concepts) if p.exists()]
        if not found:
            QMessageBox.warning(self, "No V0.10 checkpoint found",
                "Expected outputs/v0_10/perception/concept_memory_v0_8.json and/or outputs/v0_10/language_memory_v0_10.json. Finish/save a V0.10 batch first, or migrate from the CLI with explicit paths.")
            return
        self.cognitive = CognitiveSessionV11.from_memories(
            seed=self.seed,
            visual_memory=visual if visual.exists() else None,
            language_memory=language if language.exists() else None,
            concept_memory=concepts if concepts.exists() else None,
        )
        self.perception = self.cognitive.visual
        self.query_engine = self.cognitive.query
        self.visual_history = self.cognitive.visual_test_history
        self.cognitive.save("outputs/v0_11")
        self._refresh_header(); self._refresh_competence(); self._refresh_definition_table(); self._refresh_consolidation()
        QMessageBox.information(self, "Migration complete",
            f"Imported compact memory from:\n" + "\n".join(found) +
            f"\n\nvisual episodes={self.perception.learner.episode_count}\nlanguage episodes={self.cognitive.language.learner.episode_count}")

    def _save_perception(self):
        try: self.cognitive.save("outputs/v0_11")
        except Exception: pass

    def _run_visual_test_v101(self):
        try:
            rep = self.cognitive.test_visual(self.v_count.value(), float(self.v_diff.currentData()))
        except Exception as exc:
            QMessageBox.critical(self, "Visual test failed", str(exc)); return
        self.v_mem.setText(f"memory unchanged: {rep.learner_episode_count_before} → {rep.learner_episode_count_after}")
        self._fill_matrix_v101(self.v_color, rep.color_labels, rep.color_confusion)
        self._fill_matrix_v101(self.v_shape, rep.shape_labels, rep.shape_confusion)
        self.v_fail.setPlainText("\n".join(
            f"truth {x.truth_color} {x.truth_shape} → pred {x.pred_color} {x.pred_shape}"
            for x in rep.failures) or "No retained failures.")
        self.visual_history = self.cognitive.visual_test_history
        self.v_graph.set_history(self.visual_history)
        self._refresh_consolidation()

    def _run_language_test_v101(self):
        try:
            rep = self.cognitive.test_language(self.lt_count.value())
        except Exception as exc:
            QMessageBox.critical(self, "Language test failed", str(exc)); return
        self.lt_mem.setText(f"memory unchanged: {rep.learner_episode_count_before} → {rep.learner_episode_count_after}")
        self._fill_matrix_v101(self.lt_intent, rep.intent_labels, rep.intent_confusion)
        self._fill_matrix_v101(self.lt_relation, rep.relation_labels, rep.relation_confusion)
        self.lt_fail.setPlainText("\n\n".join(
            f"[{x.skill}] {x.utterance}\nEXPECTED\n{x.expected}\nPREDICTED\n{x.predicted}"
            for x in rep.failures) or "No retained failures.")
        self.lt_graph.set_history(self.cognitive.language_test_history)
        self._refresh_consolidation()

    def _language_tick(self):
        super()._language_tick()
        if hasattr(self, "c_audit"):
            self._refresh_consolidation()

    def _perception_tick(self):
        super()._perception_tick()
        if hasattr(self, "c_audit"):
            self._refresh_consolidation()

    def _refresh_header(self):
        if not hasattr(self, "memory_label"):
            return
        errors = getattr(getattr(self, "cognitive", None), "errors", None)
        ecount = len(errors.signatures) if errors is not None else 0
        self.memory_label.setText(
            f"visual {self.perception.learner.episode_count} • language {self.cognitive.language.learner.episode_count} • definitions {self.cognitive.concepts.definition_count} • error signatures {ecount}"
        )


def run_app(seed: int = 11) -> int:
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = APCNV11Window(seed)
    win.show()
    return app.exec()
