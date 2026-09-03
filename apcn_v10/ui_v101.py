from __future__ import annotations

from typing import Dict, List

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QPlainTextEdit,
)

from apcn_v08.testing_v082 import run_bulk_test
from .ui import APCNV10Window
from .ui_widgets import LearningCurve
from .ui_v101_widgets import SpikingGraph, VisualLearningCurve
from .language_v101 import CognitiveSessionV101
from .query import KnowledgeQueryEngine
from .diagnostics import language_trace, definition_trace


class APCNV101Window(APCNV10Window):
    """V0.10.1 UI patch over the stable V0.10 studio."""

    def __init__(self, seed: int = 10):
        super().__init__(seed)
        self.setWindowTitle("APCN V0.10.1 — Automatic Learning Studio")
        self.cognitive = CognitiveSessionV101(seed)
        self.query_engine = KnowledgeQueryEngine(self.cognitive.concepts)
        self.visual_history: List[Dict[str, float]] = []
        self._install_training_spike_graphs()
        if self.tabs.count() >= 4:
            self.tabs.removeTab(3)
        self.tabs.addTab(self._ask_tab_v101(), "Ask APCN")
        self.tabs.addTab(self._testing_tab_v101(), "Testing Ground")
        self._refresh_competence()
        self._refresh_definition_table()
        self._refresh_header()

    def _install_training_spike_graphs(self) -> None:
        p_frames = self.tabs.widget(0).findChildren(QFrame)
        self.p_spikes = SpikingGraph("PERCEPTION FIRING / SPIKING")
        self.p_spikes.setMaximumHeight(235)
        if p_frames and p_frames[-1].layout() is not None:
            p_frames[-1].layout().addWidget(self.p_spikes)

        l_frames = self.tabs.widget(1).findChildren(QFrame)
        self.l_spikes = SpikingGraph("LANGUAGE FIRING / SPIKING")
        if l_frames and l_frames[-1].layout() is not None:
            l_frames[-1].layout().addWidget(self.l_spikes, 1)

        d_frames = self.tabs.widget(2).findChildren(QFrame)
        self.def_spikes = SpikingGraph("DEFINITION FIRING / SPIKING")
        if d_frames and d_frames[-1].layout() is not None:
            d_frames[-1].layout().addWidget(self.def_spikes, 1)

    def _perception_tick(self):
        super()._perception_tick()
        last = getattr(self.perception, "last_step", None)
        if last is not None and hasattr(self.perception.learner, "activation_trace"):
            self.p_spikes.set_trace(
                self.perception.learner.activation_trace(last.features, last.episode.utterance)
            )

    def _language_tick(self):
        super()._language_tick()
        last = getattr(self.cognitive.language, "last_step", None)
        if last is not None:
            self.l_spikes.set_trace(
                language_trace(self.cognitive.language.learner, last.episode.utterance, last.episode.program)
            )

    def _learn_definition_curriculum(self):
        super()._learn_definition_curriculum()
        defined = [r for r in self.cognitive.concepts.records.values() if r.definition is not None]
        if defined:
            self.def_spikes.set_trace(definition_trace(self.cognitive.concepts, defined[-1].name))

    def _teach_definition(self):
        before = set(self.cognitive.concepts.records)
        super()._teach_definition()
        after = list(self.cognitive.concepts.records)
        candidates = [x for x in after if x not in before]
        name = candidates[-1] if candidates else (after[-1] if after else None)
        if name:
            self.def_spikes.set_trace(definition_trace(self.cognitive.concepts, name))

    def _ask_tab_v101(self):
        page = QWidget(); out = QVBoxLayout(page); out.setContentsMargins(10,10,10,10); out.setSpacing(8)
        note = QLabel(
            "Ask the explicit APCN concept memory. This is intentionally not a hidden external LLM: "
            "unknown concepts remain unknown, making the test falsifiable."
        ); note.setWordWrap(True); note.setStyleSheet("color:#9db0c7"); out.addWidget(note)
        self.ask_log = QPlainTextEdit(); self.ask_log.setReadOnly(True)
        self.ask_log.setPlainText(
            "Try after learning the definition curriculum:\n"
            "• what is acceleration?\n"
            "• what does force depend on?\n"
            "• do you understand pressure?\n"
            "• calculate acceleration if velocity change = 20 and time = 4"
        ); out.addWidget(self.ask_log, 1)
        line = QHBoxLayout(); self.ask_input = QLineEdit(); self.ask_input.setPlaceholderText("what is acceleration?"); self.ask_input.returnPressed.connect(self._ask_v101); ask = QPushButton("Ask APCN"); ask.setObjectName("primary"); ask.clicked.connect(self._ask_v101); line.addWidget(self.ask_input,1); line.addWidget(ask); out.addLayout(line)
        self.ask_spikes = SpikingGraph("QUERY CONCEPT ACTIVATION"); self.ask_spikes.setMaximumHeight(220); out.addWidget(self.ask_spikes)
        return page

    def _ask_v101(self):
        q = self.ask_input.text().strip()
        if not q: return
        ans = self.query_engine.ask(q)
        self.ask_log.appendPlainText(f"\nYOU: {q}\nAPCN: {ans.answer}\n")
        self.ask_input.clear()
        self.ask_spikes.set_trace(definition_trace(self.cognitive.concepts, ans.trace_concept))

    def _testing_tab_v101(self):
        page = QWidget(); out = QVBoxLayout(page); out.setContentsMargins(6,6,6,6)
        tabs = QTabWidget(); tabs.addTab(self._visual_testing_v101(), "Colors + Shapes"); tabs.addTab(self._language_testing_v101(), "Language"); out.addWidget(tabs)
        return page

    def _visual_testing_v101(self):
        page=QWidget(); out=QVBoxLayout(page); out.setContentsMargins(7,7,7,7); out.setSpacing(7)
        control=self._card(); c=QHBoxLayout(control); self.v_count=QSpinBox(); self.v_count.setRange(50,10000); self.v_count.setValue(500); self.v_diff=QComboBox(); self.v_diff.addItem("Clean",.15); self.v_diff.addItem("Normal",.55); self.v_diff.addItem("Hard",.82); self.v_diff.addItem("Stress",.95); self.v_diff.setCurrentIndex(2); run=QPushButton("Run Color + Shape Test — no learning"); run.setObjectName("primary"); run.clicked.connect(self._run_visual_test_v101); self.v_mem=QLabel("memory unchanged: —"); c.addWidget(QLabel("samples")); c.addWidget(self.v_count); c.addWidget(QLabel("difficulty")); c.addWidget(self.v_diff); c.addWidget(run); c.addStretch(); c.addWidget(self.v_mem); out.addWidget(control)
        mats=QHBoxLayout(); self.v_color=QTableWidget(); self.v_shape=QTableWidget(); mats.addWidget(self._table_card_v101("Color confusion — rows truth, columns prediction",self.v_color),1); mats.addWidget(self._table_card_v101("Shape confusion — rows truth, columns prediction",self.v_shape),1); out.addLayout(mats,1)
        bottom=QHBoxLayout(); fail=self._card(); fl=QVBoxLayout(fail); fl.addWidget(QLabel("Failures")); self.v_fail=QPlainTextEdit(); self.v_fail.setReadOnly(True); fl.addWidget(self.v_fail); graph=self._card(); gl=QVBoxLayout(graph); gl.addWidget(QLabel("Graphs")); self.v_graph=VisualLearningCurve(); gl.addWidget(self.v_graph); bottom.addWidget(fail,1); bottom.addWidget(graph,1); out.addLayout(bottom,1); return page

    def _run_visual_test_v101(self):
        try:
            rep=run_bulk_test(self.perception.learner,self.v_count.value(),float(self.v_diff.currentData()),seed=10101+37*len(self.visual_history))
        except Exception as exc:
            QMessageBox.critical(self,"Visual test failed",str(exc)); return
        self.v_mem.setText(f"memory unchanged: {rep.learner_episode_count_before} → {rep.learner_episode_count_after}")
        self._fill_matrix_v101(self.v_color,rep.color_labels,rep.color_confusion); self._fill_matrix_v101(self.v_shape,rep.shape_labels,rep.shape_confusion)
        self.v_fail.setPlainText("\n".join(f"truth {x.truth_color} {x.truth_shape} → pred {x.pred_color} {x.pred_shape}" for x in rep.failures) or "No retained failures.")
        self.visual_history.append({"episodes":float(self.perception.learner.episode_count),"color":rep.color_accuracy,"shape":rep.shape_accuracy,"joint":rep.joint_accuracy}); self.v_graph.set_history(self.visual_history)

    def _language_testing_v101(self):
        page=QWidget(); out=QVBoxLayout(page); out.setContentsMargins(7,7,7,7); out.setSpacing(7)
        control=self._card(); c=QHBoxLayout(control); self.lt_count=QSpinBox(); self.lt_count.setRange(120,5000); self.lt_count.setValue(600); run=QPushButton("Run Generated Language Test — no learning"); run.setObjectName("primary"); run.clicked.connect(self._run_language_test_v101); self.lt_mem=QLabel("memory unchanged: —"); c.addWidget(QLabel("samples")); c.addWidget(self.lt_count); c.addWidget(run); c.addStretch(); c.addWidget(self.lt_mem); out.addWidget(control)
        mats=QHBoxLayout(); self.lt_intent=QTableWidget(); self.lt_relation=QTableWidget(); mats.addWidget(self._table_card_v101("Intent confusion — rows truth, columns prediction",self.lt_intent),1); mats.addWidget(self._table_card_v101("Relation confusion — rows truth, columns prediction",self.lt_relation),1); out.addLayout(mats,1)
        bottom=QHBoxLayout(); fail=self._card(); fl=QVBoxLayout(fail); fl.addWidget(QLabel("Failures")); self.lt_fail=QPlainTextEdit(); self.lt_fail.setReadOnly(True); fl.addWidget(self.lt_fail); graph=self._card(); gl=QVBoxLayout(graph); gl.addWidget(QLabel("Graphs")); self.lt_graph=LearningCurve(); gl.addWidget(self.lt_graph); bottom.addWidget(fail,1); bottom.addWidget(graph,1); out.addLayout(bottom,1); return page

    def _run_language_test_v101(self):
        try: rep=self.cognitive.test_language(self.lt_count.value())
        except Exception as exc: QMessageBox.critical(self,"Language test failed",str(exc)); return
        self.lt_mem.setText(f"memory unchanged: {rep.learner_episode_count_before} → {rep.learner_episode_count_after}")
        self._fill_matrix_v101(self.lt_intent,rep.intent_labels,rep.intent_confusion); self._fill_matrix_v101(self.lt_relation,rep.relation_labels,rep.relation_confusion)
        self.lt_fail.setPlainText("\n\n".join(f"[{x.skill}] {x.utterance}\nEXPECTED\n{x.expected}\nPREDICTED\n{x.predicted}" for x in rep.failures) or "No retained failures.")
        self.lt_graph.set_history(self.cognitive.test_history)

    @staticmethod
    def _fill_matrix_v101(table, labels, matrix):
        table.setRowCount(len(labels)); table.setColumnCount(len(labels)); table.setHorizontalHeaderLabels(labels); table.setVerticalHeaderLabels(labels); table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for r,row in enumerate(matrix):
            for c,value in enumerate(row):
                item=QTableWidgetItem(str(value)); item.setTextAlignment(Qt.AlignmentFlag.AlignCenter); table.setItem(r,c,item)

    def _table_card_v101(self,title,table):
        card=self._card(); lay=QVBoxLayout(card); lay.setContentsMargins(6,6,6,6); lay.addWidget(QLabel(title)); lay.addWidget(table); return card


def run_app(seed: int = 10) -> int:
    import sys
    from PyQt6.QtWidgets import QApplication
    from .ui_widgets import STYLE
    app=QApplication.instance() or QApplication(sys.argv); app.setStyleSheet(STYLE); win=APCNV101Window(seed); win.show(); return app.exec()
