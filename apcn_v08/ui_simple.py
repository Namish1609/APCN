from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRectF, QPointF, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QBrush
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QProgressBar, QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QPlainTextEdit,
)

from .session import TrainingSessionV08, StepResult
from .testing_v082 import BulkTestReport, run_bulk_test


APP_STYLE = """
QWidget { background:#0b1119; color:#edf4ff; font-size:12px; }
QFrame#card { background:#121b27; border:1px solid #27364a; border-radius:9px; }
QFrame#preview { background:#10263a; border:1px solid #2879a6; border-radius:9px; }
QFrame#learn { background:#302512; border:1px solid #a87928; border-radius:9px; }
QFrame#done { background:#132d21; border:1px solid #378359; border-radius:9px; }
QPushButton { background:#1b2939; color:#fff; border:1px solid #344b67; border-radius:7px; padding:8px 12px; font-weight:700; }
QPushButton:hover { background:#24364b; }
QPushButton#primary { background:#155b7a; border-color:#2b8eb8; }
QPushButton#learnButton { background:#6b4c12; border-color:#aa7b23; }
QLineEdit,QComboBox,QSpinBox,QPlainTextEdit,QTableWidget { background:#0e1621; color:#fff; border:1px solid #2e4057; border-radius:6px; padding:5px; }
QTabWidget::pane { border:1px solid #27364a; border-radius:8px; }
QTabBar::tab { background:#101923; color:#9eb0c7; padding:8px 14px; border:1px solid #27364a; }
QTabBar::tab:selected { background:#1b2939; color:#fff; }
QProgressBar { background:#0e1621; border:1px solid #2e4057; border-radius:5px; text-align:center; min-height:17px; }
QProgressBar::chunk { background:#2d94ba; }
QHeaderView::section { background:#182333; color:#dfe9f7; border:1px solid #2d3e54; padding:4px; font-weight:700; }
"""


class SceneCanvas(QWidget):
    selectionChanged = pyqtSignal(object)

    def __init__(self, allow_selection: bool = False):
        super().__init__()
        self.image: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.title = "No example yet"
        self.subtitle = ""
        self.allow_selection = allow_selection
        self._drag_start: Optional[QPointF] = None
        self._drag_end: Optional[QPointF] = None
        self.setMinimumSize(560, 430)

    def set_scene(self, image: np.ndarray, mask: Optional[np.ndarray], title: str, subtitle: str = "") -> None:
        self.image = np.ascontiguousarray(image.copy())
        self.mask = None if mask is None else np.ascontiguousarray(mask.copy())
        self.title, self.subtitle = title, subtitle
        self._drag_start = self._drag_end = None
        self.update()

    def _image_rect(self) -> QRectF:
        avail = QRectF(14, 50, max(10, self.width()-28), max(10, self.height()-64))
        if self.image is None:
            return avail
        h, w = self.image.shape[:2]
        scale = min(avail.width()/w, avail.height()/h)
        rw, rh = w*scale, h*scale
        return QRectF(avail.x()+(avail.width()-rw)/2, avail.y()+(avail.height()-rh)/2, rw, rh)

    def _widget_to_image(self, point: QPointF) -> Optional[Tuple[int, int]]:
        if self.image is None:
            return None
        rect = self._image_rect()
        if not rect.contains(point):
            return None
        h, w = self.image.shape[:2]
        x = int(np.clip((point.x()-rect.x()) / max(1.0, rect.width()) * w, 0, w-1))
        y = int(np.clip((point.y()-rect.y()) / max(1.0, rect.height()) * h, 0, h-1))
        return x, y

    def selection_mask(self) -> Optional[np.ndarray]:
        if self.image is None:
            return None
        if self._drag_start is None or self._drag_end is None:
            return self.mask
        a, b = self._widget_to_image(self._drag_start), self._widget_to_image(self._drag_end)
        if a is None or b is None:
            return self.mask
        x1, x2 = sorted((a[0], b[0])); y1, y2 = sorted((a[1], b[1]))
        if x2-x1 < 3 or y2-y1 < 3:
            return self.mask
        out = np.zeros(self.image.shape[:2], dtype=np.uint8)
        out[y1:y2+1, x1:x2+1] = 255
        return out

    def mousePressEvent(self, event):
        if self.allow_selection and event.button() == Qt.MouseButton.LeftButton and self.image is not None and self._image_rect().contains(event.position()):
            self._drag_start = self._drag_end = event.position(); self.update()

    def mouseMoveEvent(self, event):
        if self.allow_selection and self._drag_start is not None:
            self._drag_end = event.position(); self.update()

    def mouseReleaseEvent(self, event):
        if self.allow_selection and self._drag_start is not None:
            self._drag_end = event.position(); self.selectionChanged.emit(self.selection_mask()); self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#101722"))
        p.setPen(QColor("#f3f7fd")); p.setFont(QFont("Sans Serif", 12, 700)); p.drawText(14, 22, self.title)
        p.setPen(QColor("#93a7c0")); p.setFont(QFont("Sans Serif", 9)); p.drawText(14, 41, self.subtitle[:120])
        rect = self._image_rect()
        p.setPen(QPen(QColor("#32455d"), 1)); p.setBrush(QColor("#070c12")); p.drawRoundedRect(rect, 8, 8)
        if self.image is None:
            p.setPen(QColor("#93a7c0")); p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Generate an example"); return

        h, w = self.image.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        qimg = QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format.Format_RGB888).copy()
        p.drawImage(rect, qimg)

        mask = self.selection_mask()
        if mask is not None and np.count_nonzero(mask) > 0:
            contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Critical V0.8.2 fix: the painter still had the black canvas brush active.
            # drawRect() therefore filled the attention bounding box black. Use an
            # outline-only brush so the actual generated object remains visible.
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor("#56c7ff"), 2, Qt.PenStyle.DashLine))
            sx, sy = rect.width()/w, rect.height()/h
            for c in contours:
                x, y, cw, ch = cv2.boundingRect(c)
                p.drawRect(QRectF(rect.x()+x*sx, rect.y()+y*sy, cw*sx, ch*sy))


class FiringGraph(QWidget):
    """Neuron-like APCN activity display. Nodes are concepts/features, not neural neurons."""
    def __init__(self):
        super().__init__()
        self.trace: Dict[str, object] = {"nodes": [], "edges": []}
        self.phase = 0.0
        self.timer = QTimer(self); self.timer.timeout.connect(self._pulse); self.timer.start(120)
        self.setMinimumHeight(210)

    def _pulse(self):
        self.phase += 0.55; self.update()

    def set_trace(self, trace: Dict[str, object]):
        self.trace = trace; self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); p.fillRect(self.rect(), QColor("#0b121c"))
        p.setPen(QColor("#e9f2ff")); p.setFont(QFont("Sans Serif", 10, 700)); p.drawText(10, 19, "APCN FIRING  •  concepts ↔ sensory features")
        nodes = list(self.trace.get("nodes", [])); edges = list(self.trace.get("edges", []))
        if not nodes:
            p.setPen(QColor("#8499b2")); p.drawText(QRectF(10,35,self.width()-20,self.height()-45), Qt.AlignmentFlag.AlignCenter, "No activity yet")
            return
        kinds = {"word": [], "concept": [], "family": [], "feature": []}
        for n in nodes: kinds.setdefault(str(n.get("kind","concept")), []).append(n)
        xmap = {"word": .12, "concept": .34, "family": .56, "feature": .83}
        pos: Dict[str,QPointF] = {}
        for kind in ("word","concept","family","feature"):
            arr = sorted(kinds.get(kind,[]), key=lambda n: float(n.get("firing",0)), reverse=True)[:8]
            for i,n in enumerate(arr):
                pos[str(n["id"])] = QPointF(self.width()*xmap[kind], 38+(self.height()-52)*(i+.5)/max(1,len(arr)))
        for e in edges:
            a,b = pos.get(str(e.get("src"))), pos.get(str(e.get("dst")))
            if a is None or b is None: continue
            w = float(np.clip(float(e.get("weight",.2)),0,1)); p.setPen(QPen(QColor(80,155,210,int(45+130*w)), .7+1.5*w)); p.drawLine(a,b)
        cmap = {"word":QColor("#c49cff"),"concept":QColor("#65dfa1"),"family":QColor("#ffc857"),"feature":QColor("#55c9ff")}
        for n in nodes:
            q = pos.get(str(n["id"]));
            if q is None: continue
            fire = float(np.clip(float(n.get("firing",0)),0,1)); pulse = 1.0 + .08*math.sin(self.phase + hash(str(n["id"]))%7)
            r = (5.5 + 8.5*fire)*pulse
            p.setBrush(QBrush(cmap.get(str(n.get("kind")), QColor("#65dfa1")))); p.setPen(QPen(QColor("#dce9f8"),1)); p.drawEllipse(q,r,r)
            p.setPen(QColor("#dce7f5")); p.setFont(QFont("Sans Serif",7)); p.drawText(QRectF(q.x()-42,q.y()+r+1,84,18),Qt.AlignmentFlag.AlignHCenter,str(n.get("label",""))[:14])


class APCNSimpleWindow(QMainWindow):
    def __init__(self, seed: int = 8):
        super().__init__()
        self.setWindowTitle("APCN V0.8.2 — Grounded Concept Studio")
        self.resize(1366, 768)
        self.setMinimumSize(1100, 650)
        self.session = TrainingSessionV08(seed=seed)
        self.current: Optional[StepResult] = None
        self.auto_total = self.auto_remaining = 0
        self.auto_timer = QTimer(self); self.auto_timer.timeout.connect(self._auto_tick)
        self._build_ui(); self._new_example(); self._refresh_concepts()

    def _card(self) -> QFrame:
        f=QFrame(); f.setObjectName("card"); return f

    def _build_ui(self):
        root=QWidget(); self.setCentralWidget(root); outer=QVBoxLayout(root); outer.setContentsMargins(8,6,8,6); outer.setSpacing(6)
        header=QHBoxLayout(); title=QLabel("APCN V0.8.2  •  GROUNDED CONCEPT STUDIO"); title.setFont(QFont("Sans Serif",13,800)); header.addWidget(title); header.addStretch(); self.counter=QLabel("episodes: 0"); self.counter.setStyleSheet("color:#9db0c7"); header.addWidget(self.counter); outer.addLayout(header)
        self.tabs=QTabWidget(); self.tabs.addTab(self._train_tab(),"Train"); self.tabs.addTab(self._test_tab(),"Testing Ground"); self.tabs.addTab(self._concept_tab(),"Concepts"); self.tabs.addTab(self._manual_tab(),"Real Image"); outer.addWidget(self.tabs,1)

    def _train_tab(self):
        page=QWidget(); row=QHBoxLayout(page); row.setContentsMargins(6,6,6,6); row.setSpacing(8)
        self.canvas=SceneCanvas(False); row.addWidget(self.canvas,3)
        side=QWidget(); side.setMaximumWidth(500); s=QVBoxLayout(side); s.setContentsMargins(0,0,0,0); s.setSpacing(7)
        self.mode=QFrame(); self.mode.setObjectName("preview"); ml=QVBoxLayout(self.mode); ml.setContentsMargins(10,7,10,7); self.mode_title=QLabel("TEST MODE — MEMORY FROZEN"); self.mode_title.setFont(QFont("Sans Serif",11,800)); self.mode_detail=QLabel("Clean preview. This image is not learned until you click Teach."); self.mode_detail.setWordWrap(True); ml.addWidget(self.mode_title); ml.addWidget(self.mode_detail); s.addWidget(self.mode)
        choose=self._card(); cl=QHBoxLayout(choose); cl.setContentsMargins(8,6,8,6); self.color_combo=QComboBox(); self.color_combo.addItem("random"); self.color_combo.addItems(self.session.teacher.color_words); self.shape_combo=QComboBox(); self.shape_combo.addItem("random"); self.shape_combo.addItems(self.session.teacher.shape_words); cl.addWidget(QLabel("Color")); cl.addWidget(self.color_combo); cl.addWidget(QLabel("Shape")); cl.addWidget(self.shape_combo); s.addWidget(choose)
        actions=self._card(); al=QGridLayout(actions); al.setContentsMargins(8,8,8,8); self.new_btn=QPushButton("1. New / Test Example"); self.new_btn.setObjectName("primary"); self.new_btn.clicked.connect(self._new_example); self.teach_btn=QPushButton("2. Teach Visible Example"); self.teach_btn.setObjectName("learnButton"); self.teach_btn.clicked.connect(self._teach_visible); self.batch=QSpinBox(); self.batch.setRange(10,20000); self.batch.setValue(250); self.batch.setSingleStep(50); self.auto_btn=QPushButton("3. Auto Train"); self.auto_btn.clicked.connect(self._toggle_auto); self.progress=QProgressBar(); al.addWidget(self.new_btn,0,0,1,2); al.addWidget(self.teach_btn,1,0,1,2); al.addWidget(QLabel("episodes"),2,0); al.addWidget(self.batch,2,1); al.addWidget(self.auto_btn,3,0,1,2); al.addWidget(self.progress,4,0,1,2); s.addWidget(actions)
        result=self._card(); rl=QVBoxLayout(result); rl.setContentsMargins(9,7,9,7); self.teacher_line=QLabel("Teacher: —"); self.teacher_line.setWordWrap(True); self.truth_line=QLabel("Truth: —"); self.pred_line=QLabel("APCN: —"); self.pred_line.setFont(QFont("Sans Serif",10,700)); self.check_line=QLabel("—"); rl.addWidget(self.teacher_line); rl.addWidget(self.truth_line); rl.addWidget(self.pred_line); rl.addWidget(self.check_line); s.addWidget(result)
        firing_card=self._card(); fl=QVBoxLayout(firing_card); fl.setContentsMargins(4,4,4,4); self.firing=FiringGraph(); fl.addWidget(self.firing); s.addWidget(firing_card,1)
        hint=QLabel("Training: use Auto Train. Testing: use New/Test or the Testing Ground. Background/noise variation is used internally during Auto Train on purpose."); hint.setWordWrap(True); hint.setStyleSheet("color:#8fa3bb"); s.addWidget(hint)
        row.addWidget(side,2); return page

    def _test_tab(self):
        page=QWidget(); outer=QVBoxLayout(page); outer.setContentsMargins(8,8,8,8); outer.setSpacing(8)
        controls=self._card(); c=QHBoxLayout(controls); c.setContentsMargins(10,7,10,7); self.test_count=QSpinBox(); self.test_count.setRange(30,10000); self.test_count.setValue(500); self.test_count.setSingleStep(100); self.test_diff=QComboBox(); self.test_diff.addItem("Clean  0.15",.15); self.test_diff.addItem("Normal  0.55",.55); self.test_diff.addItem("Hard  0.82",.82); self.test_diff.addItem("Stress  0.95",.95); self.test_diff.setCurrentIndex(2); run=QPushButton("Run Bulk Test — no learning"); run.setObjectName("primary"); run.clicked.connect(self._bulk_test); c.addWidget(QLabel("samples")); c.addWidget(self.test_count); c.addWidget(QLabel("difficulty")); c.addWidget(self.test_diff); c.addWidget(run); c.addStretch(); self.test_memory=QLabel("memory unchanged: —"); c.addWidget(self.test_memory); outer.addWidget(controls)
        scores=QFrame(); scores.setObjectName("card"); sg=QGridLayout(scores); self.score_color=QLabel("COLOR\n—"); self.score_shape=QLabel("SHAPE\n—"); self.score_joint=QLabel("JOINT\n—");
        for i,w in enumerate((self.score_color,self.score_shape,self.score_joint)):
            w.setAlignment(Qt.AlignmentFlag.AlignCenter); w.setFont(QFont("Sans Serif",16,800)); sg.addWidget(w,0,i)
        outer.addWidget(scores)
        matrices=QHBoxLayout(); self.color_table=QTableWidget(); self.shape_table=QTableWidget(); matrices.addWidget(self._table_card("Color confusion — rows truth, columns prediction",self.color_table),1); matrices.addWidget(self._table_card("Shape confusion — rows truth, columns prediction",self.shape_table),1); outer.addLayout(matrices,1)
        bottom=QHBoxLayout(); recall=self._card(); rlay=QVBoxLayout(recall); rlay.addWidget(QLabel("Per-class recall")); self.recall_text=QPlainTextEdit(); self.recall_text.setReadOnly(True); rlay.addWidget(self.recall_text); fails=self._card(); flay=QVBoxLayout(fails); flay.addWidget(QLabel("Sample failures")); self.fail_text=QPlainTextEdit(); self.fail_text.setReadOnly(True); flay.addWidget(self.fail_text); bottom.addWidget(recall,1); bottom.addWidget(fails,1); outer.addLayout(bottom,1); return page

    def _table_card(self,title:str,table:QTableWidget):
        card=self._card(); l=QVBoxLayout(card); l.setContentsMargins(7,7,7,7); lab=QLabel(title); lab.setFont(QFont("Sans Serif",9,700)); l.addWidget(lab); l.addWidget(table); return card

    def _concept_tab(self):
        page=QWidget(); row=QHBoxLayout(page); row.setContentsMargins(8,8,8,8); left=self._card(); ll=QVBoxLayout(left); ll.addWidget(QLabel("Learned concepts")); self.concepts=QPlainTextEdit(); self.concepts.setReadOnly(True); ll.addWidget(self.concepts); right=self._card(); rl=QVBoxLayout(right); rl.addWidget(QLabel("Discovered families / memory")); self.families=QPlainTextEdit(); self.families.setReadOnly(True); rl.addWidget(self.families); buttons=QHBoxLayout(); sv=QPushButton("Save"); sv.clicked.connect(self._save); ld=QPushButton("Load"); ld.clicked.connect(self._load); buttons.addWidget(sv); buttons.addWidget(ld); rl.addLayout(buttons); row.addWidget(left,1); row.addWidget(right,1); return page

    def _manual_tab(self):
        page=QWidget(); row=QHBoxLayout(page); row.setContentsMargins(8,8,8,8); self.real_canvas=SceneCanvas(True); row.addWidget(self.real_canvas,3); side=self._card(); side.setMaximumWidth(430); s=QVBoxLayout(side); h=QLabel("Real-image teaching"); h.setFont(QFont("Sans Serif",12,800)); s.addWidget(h); info=QLabel("Load an image, drag a box around the object, type what you want to teach, then teach the focus. This updates the same memory as synthetic training."); info.setWordWrap(True); s.addWidget(info); load=QPushButton("Load image"); load.clicked.connect(self._load_real); s.addWidget(load); self.manual_text=QLineEdit(); self.manual_text.setPlaceholderText("this is a yellow ball"); s.addWidget(self.manual_text); teach=QPushButton("Teach focused object"); teach.setObjectName("learnButton"); teach.clicked.connect(self._teach_real); s.addWidget(teach); self.manual_status=QLabel("No real image loaded."); self.manual_status.setWordWrap(True); s.addWidget(self.manual_status); s.addStretch(); row.addWidget(side,2); return page

    def _choice(self,combo): return None if combo.currentText()=="random" else combo.currentText()

    def _set_mode(self,kind,title,detail):
        self.mode.setObjectName(kind); self.mode.style().unpolish(self.mode); self.mode.style().polish(self.mode); self.mode_title.setText(title); self.mode_detail.setText(detail)

    def _display(self,r:StepResult,training_sample:bool=False):
        self.current=r; ep=r.episode; tc=str(ep.teacher_metadata.get("color","?")); ts=str(ep.teacher_metadata.get("shape","?")); pc=r.predicted_color or "?"; ps=r.predicted_shape or "?"
        subtitle = "actual automatic-training sample • nuisance variation may be present" if training_sample else "clean prediction-only sample • dashed outline = teacher focus"
        self.canvas.set_scene(ep.image,ep.attention_mask,f"Visible example: {tc} {ts}",subtitle)
        self.teacher_line.setText(f"Teacher:  {ep.utterance}"); self.truth_line.setText(f"Truth:  {tc} {ts}"); self.pred_line.setText(f"APCN:  {pc} {ps}")
        marks=f"{'✓' if pc==tc else '✗'} color   {'✓' if ps==ts else '✗'} shape"; self.check_line.setText(marks); self.check_line.setStyleSheet("color:#69e39c" if pc==tc and ps==ts else "color:#ffc857")
        self.firing.set_trace(self.session.learner.activation_trace(r.features,ep.utterance)); self.counter.setText(f"episodes: {self.session.learner.episode_count}  •  vocabulary: {len(self.session.learner.token_stats)}"); self._refresh_concepts()

    def _new_example(self):
        r=self.session.generate_preview(self._choice(self.color_combo),self._choice(self.shape_combo),difficulty=0.0,add_distractors=False); self._set_mode("preview","TEST MODE — MEMORY FROZEN","Clean example for visual verification. Click Teach only if you want this exact example added to memory."); self._display(r,False)

    def _teach_visible(self):
        if self.current is None: return
        ep=self.current.episode; self.session.teach_current(ep.utterance,ep.image,ep.attention_mask); self._set_mode("done","LEARNED ONE EXAMPLE","The visible image + teacher sentence was added once. Generate a new example to test generalization."); self.counter.setText(f"episodes: {self.session.learner.episode_count}  •  vocabulary: {len(self.session.learner.token_stats)}"); self.firing.set_trace(self.session.learner.activation_trace(self.current.features,ep.utterance)); self._refresh_concepts()

    def _toggle_auto(self):
        if self.auto_timer.isActive(): self.auto_timer.stop(); self.auto_btn.setText("3. Auto Train"); self._set_mode("preview","PAUSED","Automatic training paused; memory already contains completed episodes."); return
        self.auto_total=self.batch.value(); self.auto_remaining=self.auto_total; self.progress.setValue(0); self.auto_btn.setText("Pause Auto Train"); self._set_mode("learn","AUTO TRAINING — MEMORY CHANGING",f"0 / {self.auto_total} generated lessons. Training examples intentionally vary background, lighting, position and clutter."); self.auto_timer.start(1)

    def _auto_tick(self):
        if self.auto_remaining<=0:
            self.auto_timer.stop(); self.auto_btn.setText("3. Auto Train"); self.progress.setValue(100); self._set_mode("done","AUTO TRAIN COMPLETE",f"Finished {self.auto_total} lessons. Use New/Test or Testing Ground to measure generalization."); self.counter.setText(f"episodes: {self.session.learner.episode_count}  •  vocabulary: {len(self.session.learner.token_stats)}"); self._refresh_concepts(); return
        chunk=min(5,self.auto_remaining); last=None
        for _ in range(chunk): last=self.session.step()
        self.auto_remaining-=chunk; done=self.auto_total-self.auto_remaining; self.progress.setValue(int(100*done/max(1,self.auto_total))); self.mode_detail.setText(f"{done} / {self.auto_total} lessons learned. Nuisance variation is ON during training.")
        if last is not None and (done%50==0 or self.auto_remaining==0): self._display(last,True)

    def _bulk_test(self):
        n=self.test_count.value(); d=float(self.test_diff.currentData()); self.test_memory.setText("running…"); QApplication.processEvents()
        try: rep=run_bulk_test(self.session.learner,n,d,seed=28082+n+int(d*1000))
        except Exception as exc: QMessageBox.critical(self,"Bulk test failed",str(exc)); self.test_memory.setText("failed"); return
        self._render_report(rep)

    def _render_report(self,rep:BulkTestReport):
        self.score_color.setText(f"COLOR\n{rep.color_accuracy:.1%}"); self.score_shape.setText(f"SHAPE\n{rep.shape_accuracy:.1%}"); self.score_joint.setText(f"JOINT\n{rep.joint_accuracy:.1%}"); self.test_memory.setText(f"memory unchanged: {rep.learner_episode_count_before} → {rep.learner_episode_count_after}"); self._fill_matrix(self.color_table,rep.color_labels,rep.color_confusion); self._fill_matrix(self.shape_table,rep.shape_labels,rep.shape_confusion)
        lines=["COLORS"]+[f"  {k:<10} {v:.1%}" for k,v in rep.color_recall.items()]+["","SHAPES"]+[f"  {k:<10} {v:.1%}" for k,v in rep.shape_recall.items()]; self.recall_text.setPlainText("\n".join(lines)); self.fail_text.setPlainText("\n".join(f"truth {f.truth_color} {f.truth_shape:<10} → pred {f.pred_color} {f.pred_shape}" for f in rep.failures) or "No failures in retained sample.")

    def _fill_matrix(self,table:QTableWidget,labels,mat):
        table.setRowCount(len(labels)); table.setColumnCount(len(labels)); table.setHorizontalHeaderLabels(labels); table.setVerticalHeaderLabels(labels)
        for r,row in enumerate(mat):
            for c,val in enumerate(row):
                item=QTableWidgetItem(str(val)); item.setTextAlignment(Qt.AlignmentFlag.AlignCenter); table.setItem(r,c,item)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def _refresh_concepts(self):
        if not hasattr(self,"concepts"): return
        rows=[]
        for t,s in sorted(self.session.learner.token_stats.items(),key=lambda kv:self.session.learner.concept_quality(kv[0]),reverse=True): rows.append(f"{t:<13} q={self.session.learner.concept_quality(t):.3f}  support={s.count:<5d}  {self.session.learner.role_guess(t)}")
        self.concepts.setPlainText("\n".join(rows) if rows else "No learned concepts yet."); fams=self.session.learner.discover_families(); self.families.setPlainText("\n\n".join(f"{f['id']}  similarity={f['mean_similarity']:.3f}\n  "+", ".join(f['members']) for f in fams) if fams else "No stable families yet.")

    def _save(self):
        path=self.session.save(); QMessageBox.information(self,"Saved",f"Memory saved to\n{path}")

    def _load(self):
        path,_=QFileDialog.getOpenFileName(self,"Load APCN memory",str(Path("outputs/v0_8/concept_memory_v0_8.json")),"JSON (*.json)")
        if not path:return
        try:self.session=TrainingSessionV08.load(path)
        except Exception as exc: QMessageBox.critical(self,"Load failed",str(exc)); return
        self.color_combo.clear(); self.color_combo.addItem("random"); self.color_combo.addItems(self.session.teacher.color_words); self.shape_combo.clear(); self.shape_combo.addItem("random"); self.shape_combo.addItems(self.session.teacher.shape_words); self._refresh_concepts(); self._new_example()

    def _load_real(self):
        path,_=QFileDialog.getOpenFileName(self,"Load image","","Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path:return
        img=cv2.imread(path,cv2.IMREAD_COLOR)
        if img is None: QMessageBox.critical(self,"Image error","Could not read image."); return
        self.real_canvas.set_scene(img,None,"Real image","Drag a rectangle around the object to define joint attention."); self.manual_status.setText("Image loaded. Draw a focus box, type a sentence, then teach.")

    def _teach_real(self):
        text=self.manual_text.text().strip(); mask=self.real_canvas.selection_mask()
        if self.real_canvas.image is None or mask is None or np.count_nonzero(mask)<8: QMessageBox.information(self,"Focus required","Load an image and draw a focus rectangle first."); return
        if not text: QMessageBox.information(self,"Sentence required","Type a teaching sentence first."); return
        try:x=self.session.teach_current(text,self.real_canvas.image,mask)
        except Exception as exc: QMessageBox.critical(self,"Teaching failed",str(exc)); return
        self.manual_status.setText(f"Learned once: {text}\nTotal episodes: {self.session.learner.episode_count}"); self.counter.setText(f"episodes: {self.session.learner.episode_count}  •  vocabulary: {len(self.session.learner.token_stats)}"); self.firing.set_trace(self.session.learner.activation_trace(x,text)); self._refresh_concepts()


def run_app(seed:int=8)->int:
    app=QApplication.instance() or QApplication([]); app.setStyle("Fusion"); app.setStyleSheet(APP_STYLE); win=APCNSimpleWindow(seed); win.show(); return app.exec()
