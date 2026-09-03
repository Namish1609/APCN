from __future__ import annotations

from typing import List
import json
import time
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QProgressBar, QSpinBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QPlainTextEdit,
)

from apcn_v08.session import TrainingSessionV08
from .definitions import DefinitionParseError
from .session import CognitiveSessionV10
from .ui_widgets import STYLE, ImageCanvas, LearningCurve


class APCNV10Window(QMainWindow):
    def __init__(self, seed: int = 10):
        super().__init__()
        self.setWindowTitle("APCN V0.10 — Automatic Learning Studio")
        self.resize(1366, 768)
        self.setMinimumSize(1120, 650)
        self.seed = seed
        self.perception = TrainingSessionV08(seed=seed)
        self.cognitive = CognitiveSessionV10(seed=seed)

        self.p_total = self.p_remaining = 0
        self.p_timer = QTimer(self); self.p_timer.setInterval(24); self.p_timer.timeout.connect(self._perception_tick)
        self.p_started_at = 0.0
        self.l_total = self.l_remaining = 0
        self.l_timer = QTimer(self); self.l_timer.setInterval(24); self.l_timer.timeout.connect(self._language_tick)
        self.l_started_at = 0.0

        self._build_ui()
        self._refresh_definition_table()
        self._refresh_competence()
        self._refresh_header()

    def _card(self) -> QFrame:
        f = QFrame(); f.setObjectName("card"); return f

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        outer = QVBoxLayout(root); outer.setContentsMargins(8,6,8,6); outer.setSpacing(6)
        hdr = QHBoxLayout(); title=QLabel("APCN V0.10  •  AUTOMATIC GROUNDED LANGUAGE + CONCEPT LEARNING"); title.setFont(QFont("Sans Serif",13,800)); hdr.addWidget(title); hdr.addStretch(); self.memory_label=QLabel("visual 0 • language 0 • definitions 0"); self.memory_label.setStyleSheet("color:#9db0c7"); hdr.addWidget(self.memory_label); outer.addLayout(hdr)
        self.tabs=QTabWidget(); self.tabs.addTab(self._perception_tab(),"Perception"); self.tabs.addTab(self._language_tab(),"Language"); self.tabs.addTab(self._definitions_tab(),"Definitions"); self.tabs.addTab(self._testing_tab(),"Testing Ground"); outer.addWidget(self.tabs,1)

    def _perception_tab(self):
        page=QWidget(); row=QHBoxLayout(page); row.setContentsMargins(6,6,6,6); row.setSpacing(8)
        self.p_canvas=ImageCanvas(); row.addWidget(self.p_canvas,3)
        side=self._card(); side.setMaximumWidth(500); s=QVBoxLayout(side); s.setContentsMargins(10,9,10,9); s.setSpacing(8)
        h=QLabel("Automatic shape + color grounding"); h.setFont(QFont("Sans Serif",12,800)); s.addWidget(h)
        info=QLabel("The curriculum varies color, shape, position, lighting and clutter automatically. Progress below is live; pause/resume never discards the unfinished batch."); info.setWordWrap(True); info.setStyleSheet("color:#9db0c7"); s.addWidget(info)
        ctrl=QHBoxLayout(); self.p_batch=QSpinBox(); self.p_batch.setRange(50,20000); self.p_batch.setValue(500); self.p_batch.setSingleStep(100); self.p_button=QPushButton("Start Auto Train"); self.p_button.setObjectName("primary"); self.p_button.clicked.connect(self._toggle_perception); ctrl.addWidget(QLabel("experiences")); ctrl.addWidget(self.p_batch); ctrl.addWidget(self.p_button,1); s.addLayout(ctrl)
        self.p_progress=QProgressBar(); self.p_progress.setRange(0,1000); self.p_progress.setValue(0); self.p_progress.setFormat("0.0%") ; s.addWidget(self.p_progress)
        self.p_status=QLabel("0 / 500 complete • 500 remaining • 0.0%\nphase: waiting"); self.p_status.setWordWrap(True); self.p_status.setFont(QFont("Sans Serif",10,700)); s.addWidget(self.p_status)
        self.p_current=QLabel("Current sample: —"); self.p_current.setWordWrap(True); s.addWidget(self.p_current)
        self.p_hint=QLabel("Training starts slowly enough to see progress but automatically increases the number of experiences processed per UI tick for large batches."); self.p_hint.setWordWrap(True); self.p_hint.setStyleSheet("color:#8fa3bb"); s.addWidget(self.p_hint)
        s.addStretch(); row.addWidget(side,2); return page

    def _language_tab(self):
        page=QWidget(); row=QHBoxLayout(page); row.setContentsMargins(7,7,7,7); row.setSpacing(8)
        left=self._card(); ll=QVBoxLayout(left); ll.setContentsMargins(10,9,10,9); ll.setSpacing(8)
        h=QLabel("Automatic grounded semantic language acquisition"); h.setFont(QFont("Sans Serif",12,800)); ll.addWidget(h)
        info=QLabel("No relation/group/sequence selector. APCN measures its own competence and automatically chooses the weakest unlocked skill while keeping training evidence balanced."); info.setWordWrap(True); info.setStyleSheet("color:#9db0c7"); ll.addWidget(info)
        ctrl=QHBoxLayout(); self.l_batch=QSpinBox(); self.l_batch.setRange(100,50000); self.l_batch.setValue(1200); self.l_batch.setSingleStep(200); self.l_button=QPushButton("Start Language Learning"); self.l_button.setObjectName("primary"); self.l_button.clicked.connect(self._toggle_language); ctrl.addWidget(QLabel("experiences")); ctrl.addWidget(self.l_batch); ctrl.addWidget(self.l_button,1); ll.addLayout(ctrl)
        self.l_progress=QProgressBar(); self.l_progress.setRange(0,1000); self.l_progress.setValue(0); self.l_progress.setFormat("0.0%"); ll.addWidget(self.l_progress)
        self.l_status=QLabel("0 / 1200 complete • 1200 remaining • 0.0%\ncurrent skill: waiting"); self.l_status.setFont(QFont("Sans Serif",10,700)); self.l_status.setWordWrap(True); ll.addWidget(self.l_status)
        self.l_sentence=QLabel("Teacher sentence: —"); self.l_sentence.setWordWrap(True); ll.addWidget(self.l_sentence)
        programs=QHBoxLayout(); self.l_expected=QPlainTextEdit(); self.l_predicted=QPlainTextEdit(); self.l_expected.setReadOnly(True); self.l_predicted.setReadOnly(True); programs.addWidget(self._labeled("World meaning", self.l_expected),1); programs.addWidget(self._labeled("APCN before learning", self.l_predicted),1); ll.addLayout(programs,1)
        row.addWidget(left,3)
        right=self._card(); right.setMaximumWidth(420); rl=QVBoxLayout(right); rl.setContentsMargins(10,9,10,9); ch=QLabel("Curriculum competence"); ch.setFont(QFont("Sans Serif",11,800)); rl.addWidget(ch); self.competence=QTableWidget(); self.competence.setColumnCount(3); self.competence.setHorizontalHeaderLabels(["Skill","Evidence","Recent competence"]); self.competence.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); self.competence.verticalHeader().setVisible(False); rl.addWidget(self.competence); note=QLabel("These are internal curriculum skills. You do not choose them. Low competence causes more targeted experience, but exposure balancing prevents one failed skill from taking over the entire dataset."); note.setWordWrap(True); note.setStyleSheet("color:#8fa3bb"); rl.addWidget(note); row.addWidget(right,2); return page

    def _definitions_tab(self):
        page=QWidget(); outer=QVBoxLayout(page); outer.setContentsMargins(8,8,8,8); outer.setSpacing(8)
        top=self._card(); tl=QHBoxLayout(top); tl.setContentsMargins(10,8,10,8); self.def_button=QPushButton("Learn V0.10 definition curriculum"); self.def_button.setObjectName("primary"); self.def_button.clicked.connect(self._learn_definition_curriculum); self.def_input=QLineEdit(); self.def_input.setPlaceholderText("e.g. energy ratio is work divided by time"); self.def_input.returnPressed.connect(self._teach_definition); teach=QPushButton("Teach definition"); teach.clicked.connect(self._teach_definition); tl.addWidget(self.def_button); tl.addWidget(self.def_input,1); tl.addWidget(teach); outer.addWidget(top)
        mid=QHBoxLayout(); table_card=self._card(); tcl=QVBoxLayout(table_card); tcl.addWidget(QLabel("Concept dependency memory")); self.def_table=QTableWidget(); self.def_table.setColumnCount(5); self.def_table.setHorizontalHeaderLabels(["Concept","Kind","Definition","Dependencies","Grounding audit"]); self.def_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); self.def_table.verticalHeader().setVisible(False); tcl.addWidget(self.def_table); mid.addWidget(table_card,3)
        audit=self._card(); audit.setMaximumWidth(410); al=QVBoxLayout(audit); ah=QLabel("Concept-from-concept status"); ah.setFont(QFont("Sans Serif",11,800)); al.addWidget(ah); self.def_status=QPlainTextEdit(); self.def_status.setReadOnly(True); self.def_status.setPlainText("V0.10 can create definitions from existing concepts, execute arithmetic derived quantities, and explicitly report unresolved dependencies.\n\nIt does not claim that primitives such as mass/time are already fully grounded scientifically; those are inputs from perception/measurement layers."); al.addWidget(self.def_status); mid.addWidget(audit,2); outer.addLayout(mid,1); return page

    def _testing_tab(self):
        page=QWidget(); outer=QVBoxLayout(page); outer.setContentsMargins(8,8,8,8); outer.setSpacing(8)
        controls=self._card(); c=QHBoxLayout(controls); c.setContentsMargins(10,7,10,7); self.test_count=QSpinBox(); self.test_count.setRange(120,5000); self.test_count.setValue(600); self.test_count.setSingleStep(120); run=QPushButton("Run Generated Language Test — no learning"); run.setObjectName("primary"); run.clicked.connect(self._run_language_test); c.addWidget(QLabel("generated samples")); c.addWidget(self.test_count); c.addWidget(run); c.addStretch(); self.test_memory=QLabel("memory unchanged: —"); c.addWidget(self.test_memory); outer.addWidget(controls)
        matrices=QHBoxLayout(); self.intent_table=QTableWidget(); self.relation_table=QTableWidget(); matrices.addWidget(self._table_card("Intent confusion — rows truth, columns prediction",self.intent_table),1); matrices.addWidget(self._table_card("Relation confusion — rows truth, columns prediction",self.relation_table),1); outer.addLayout(matrices,1)
        bottom=QHBoxLayout(); fail=self._card(); fl=QVBoxLayout(fail); fl.addWidget(QLabel("Failures")); self.failures=QPlainTextEdit(); self.failures.setReadOnly(True); fl.addWidget(self.failures); graph=self._card(); gl=QVBoxLayout(graph); gl.addWidget(QLabel("Graphs")); self.graph=LearningCurve(); gl.addWidget(self.graph); bottom.addWidget(fail,1); bottom.addWidget(graph,1); outer.addLayout(bottom,1); return page

    def _labeled(self, title: str, widget: QWidget) -> QWidget:
        w=QWidget(); l=QVBoxLayout(w); l.setContentsMargins(0,0,0,0); lab=QLabel(title); lab.setStyleSheet("color:#9eb0c7"); l.addWidget(lab); l.addWidget(widget); return w

    def _table_card(self,title:str,table:QTableWidget):
        card=self._card(); l=QVBoxLayout(card); l.setContentsMargins(7,7,7,7); lab=QLabel(title); lab.setFont(QFont("Sans Serif",9,700)); l.addWidget(lab); l.addWidget(table); return card

    def _set_progress(self, bar: QProgressBar, completed: int, total: int):
        pct = 100.0*completed/max(1,total); bar.setValue(int(round(pct*10))); bar.setFormat(f"{pct:.1f}%")

    def _toggle_perception(self):
        if self.p_timer.isActive():
            self.p_timer.stop(); self.p_button.setText("Resume Auto Train"); return
        if self.p_remaining <= 0:
            self.p_total=self.p_batch.value(); self.p_remaining=self.p_total; self.p_started_at=time.monotonic(); self._set_progress(self.p_progress,0,self.p_total)
        self.p_button.setText("Pause"); self.p_timer.start()

    def _perception_tick(self):
        if self.p_remaining <= 0:
            self.p_timer.stop(); self.p_button.setText("Start Auto Train"); self._set_progress(self.p_progress,self.p_total,self.p_total); self._save_perception(); return
        chunk=max(1,self.p_total//450); chunk=min(chunk,self.p_remaining); last=None
        for _ in range(chunk): last=self.perception.step()
        self.p_remaining-=chunk; done=self.p_total-self.p_remaining; self._set_progress(self.p_progress,done,self.p_total)
        if last is not None:
            ep=last.episode; tc=ep.teacher_metadata.get("color","?"); ts=ep.teacher_metadata.get("shape","?"); phase=last.state.phase
            self.p_canvas.set_scene(ep.image,ep.attention_mask,f"Training sample: {tc} {ts}",f"phase={phase} • visual memory is changing")
            self.p_current.setText(f"Current sample: {tc} {ts}\nAPCN predicts: {last.predicted_color or '?'} {last.predicted_shape or '?'}")
            pct=100.0*done/max(1,self.p_total); self.p_status.setText(f"{done} / {self.p_total} complete • {self.p_remaining} remaining • {pct:.1f}%\nphase: {phase} • total visual episodes: {self.perception.learner.episode_count}")
        self._refresh_header()

    def _save_perception(self):
        try: self.perception.save("outputs/v0_10/perception")
        except Exception: pass

    def _toggle_language(self):
        if self.l_timer.isActive():
            self.l_timer.stop(); self.l_button.setText("Resume Language Learning"); return
        if self.l_remaining <= 0:
            self.l_total=self.l_batch.value(); self.l_remaining=self.l_total; self.l_started_at=time.monotonic(); self._set_progress(self.l_progress,0,self.l_total)
        self.l_button.setText("Pause"); self.l_timer.start()

    def _language_tick(self):
        if self.l_remaining <= 0:
            self.l_timer.stop(); self.l_button.setText("Start Language Learning"); self._set_progress(self.l_progress,self.l_total,self.l_total); self.cognitive.save(); return
        target=max(1,self.l_total//450); target=min(target,self.l_remaining); added=0; last=None
        while added < target and self.l_remaining > 0:
            before=self.cognitive.language.learner.episode_count; last=self.cognitive.language.step(); delta=max(1,self.cognitive.language.learner.episode_count-before); added += delta; self.l_remaining=max(0,self.l_remaining-delta)
        done=self.l_total-self.l_remaining; self._set_progress(self.l_progress,done,self.l_total)
        if last is not None:
            pct=100.0*done/max(1,self.l_total); self.l_status.setText(f"{done} / {self.l_total} complete • {self.l_remaining} remaining • {pct:.1f}%\ncurrent skill: {last.skill} • total language episodes: {self.cognitive.language.learner.episode_count}")
            self.l_sentence.setText(f"Teacher sentence: {last.episode.utterance}")
            self.l_expected.setPlainText(last.episode.program.pretty()); self.l_predicted.setPlainText("NO PARSE" if last.prediction is None else last.prediction.pretty())
            self._refresh_competence()
        self._refresh_header()

    def _refresh_competence(self):
        skills=self.cognitive.language.skills; self.competence.setRowCount(len(skills))
        for r,(name,st) in enumerate(skills.items()):
            for c,val in enumerate((name,str(st.attempts),f"{st.ema:.1%}")):
                item=QTableWidgetItem(val); item.setTextAlignment(Qt.AlignmentFlag.AlignCenter); self.competence.setItem(r,c,item)

    def _learn_definition_curriculum(self):
        try:
            records=self.cognitive.definitions.train_all_once()
        except Exception as exc:
            QMessageBox.critical(self,"Definition training failed",str(exc)); return
        self.def_status.setPlainText("Learned/reinforced:\n\n"+"\n".join(f"{r.name}: {r.definition.pretty() if r.definition else 'primitive'}" for r in records)+"\n\nEvery definition is constructed from existing concept references; unresolved dependencies remain explicit.")
        self._refresh_definition_table(); self._refresh_header(); self.cognitive.save()

    def _teach_definition(self):
        text=self.def_input.text().strip()
        if not text: return
        try: rec=self.cognitive.concepts.learn_definition(text)
        except DefinitionParseError as exc:
            QMessageBox.warning(self,"Definition not learned",str(exc)); return
        self.def_input.clear(); audit=self.cognitive.concepts.understanding(rec.name); self.def_status.setPlainText(json.dumps(audit,indent=2)); self._refresh_definition_table(); self._refresh_header()

    def _refresh_definition_table(self):
        rows=sorted(self.cognitive.concepts.records.values(), key=lambda r:(not r.primitive,r.name)); self.def_table.setRowCount(len(rows))
        for i,r in enumerate(rows):
            audit=self.cognitive.concepts.understanding(r.name); vals=(r.name,r.kind,"—" if r.definition is None else r.definition.pretty(),", ".join(sorted(r.dependencies())) or "—","complete" if audit.get("complete") else "missing: "+", ".join(audit.get("unresolved",[])))
            for j,val in enumerate(vals): self.def_table.setItem(i,j,QTableWidgetItem(str(val)))

    def _run_language_test(self):
        self.test_memory.setText("running generated held-out test…"); QApplication.processEvents()
        try: rep=self.cognitive.test_language(self.test_count.value())
        except Exception as exc: QMessageBox.critical(self,"Language test failed",str(exc)); self.test_memory.setText("failed"); return
        self.test_memory.setText(f"memory unchanged: {rep.learner_episode_count_before} → {rep.learner_episode_count_after}")
        self._fill_matrix(self.intent_table,rep.intent_labels,rep.intent_confusion); self._fill_matrix(self.relation_table,rep.relation_labels,rep.relation_confusion)
        text=[]
        for f in rep.failures:
            text.append(f"[{f.skill}] {f.utterance}\nEXPECTED\n{f.expected}\nPREDICTED\n{f.predicted}\n")
        self.failures.setPlainText("\n".join(text) if text else "No retained failures.")
        self.graph.set_history(self.cognitive.test_history)

    def _fill_matrix(self,table:QTableWidget,labels:List[str],matrix:List[List[int]]):
        table.setRowCount(len(labels)); table.setColumnCount(len(labels)); table.setVerticalHeaderLabels(labels); table.setHorizontalHeaderLabels(labels); table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for r,row in enumerate(matrix):
            for c,val in enumerate(row):
                item=QTableWidgetItem(str(val)); item.setTextAlignment(Qt.AlignmentFlag.AlignCenter); table.setItem(r,c,item)

    def _refresh_header(self):
        self.memory_label.setText(f"visual {self.perception.learner.episode_count} • language {self.cognitive.language.learner.episode_count} • definitions {self.cognitive.concepts.definition_count}")


def run_app(seed: int = 10) -> int:
    app=QApplication.instance() or QApplication([]); app.setStyle("Fusion"); app.setStyleSheet(STYLE); win=APCNV10Window(seed); win.show(); return app.exec()
