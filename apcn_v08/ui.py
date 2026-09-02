from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRectF, QPointF, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QBrush
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QPlainTextEdit,
    QProgressBar, QSlider, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from .session import TrainingSessionV08, StepResult

PANEL = QColor("#121a27")
TEXT = QColor("#eef4ff")
MUTED = QColor("#9fb0c9")
ACCENT = QColor("#54d2ff")
GOOD = QColor("#61e294")
WARN = QColor("#ffc857")
PURPLE = QColor("#c297ff")

APP_STYLE = """
QWidget { background:#0c111b; color:#eef4ff; font-size:13px; }
QGroupBox { border:1px solid #2a3850; border-radius:8px; margin-top:10px; padding-top:9px; font-weight:700; }
QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 5px; color:#b9c9e2; }
QLineEdit,QPlainTextEdit,QComboBox,QSpinBox { background:#101827; color:#fff; border:1px solid #344662; border-radius:6px; padding:6px; selection-background-color:#2a82a2; }
QComboBox QAbstractItemView { background:#101827; color:#fff; selection-background-color:#2a82a2; }
QPushButton { background:#1c2b40; color:#fff; border:1px solid #3b516f; border-radius:7px; padding:7px 12px; font-weight:650; }
QPushButton:hover { background:#243650; } QPushButton:checked { background:#14536b; border-color:#54d2ff; }
QTabWidget::pane { border:1px solid #2a3850; }
QTabBar::tab { background:#111a29; color:#aebed6; padding:8px 12px; border:1px solid #2a3850; }
QTabBar::tab:selected { background:#1a2a40; color:#fff; }
QProgressBar { background:#101827; border:1px solid #344662; border-radius:5px; text-align:center; color:white; }
QProgressBar::chunk { background:#2e9fc5; border-radius:4px; }
QSlider::groove:horizontal { height:5px; background:#28384f; } QSlider::handle:horizontal { width:16px; margin:-6px 0; background:#54d2ff; border-radius:8px; }
"""


class ImageCanvas(QWidget):
    selectionChanged = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.image: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.title = "TEACHER / CAMERA VIEW"
        self.subtitle = "Generate a lesson or load an image."
        self.allow_selection = False
        self._drag_start: Optional[QPointF] = None
        self._drag_end: Optional[QPointF] = None
        self.setMinimumSize(620, 440)

    def set_scene(self, image: np.ndarray, mask: Optional[np.ndarray], title: str, subtitle: str = "") -> None:
        self.image = np.ascontiguousarray(image.copy())
        self.mask = None if mask is None else np.ascontiguousarray(mask.copy())
        self.title, self.subtitle = title, subtitle
        self._drag_start = self._drag_end = None
        self.update()

    def _image_rect(self) -> QRectF:
        if self.image is None:
            return QRectF(20, 62, max(10, self.width()-40), max(10, self.height()-92))
        h, w = self.image.shape[:2]
        avail = QRectF(20, 62, max(10, self.width()-40), max(10, self.height()-92))
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
        x = int(np.clip((point.x()-rect.x())/rect.width()*w, 0, w-1))
        y = int(np.clip((point.y()-rect.y())/rect.height()*h, 0, h-1))
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
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); p.fillRect(self.rect(), PANEL)
        p.setPen(TEXT); p.setFont(QFont("Sans Serif", 13, 700)); p.drawText(18, 28, self.title)
        p.setPen(MUTED); p.setFont(QFont("Sans Serif", 9)); p.drawText(18, 47, self.subtitle[:110])
        rect = self._image_rect(); p.setPen(QPen(QColor("#30425d"), 1)); p.setBrush(QColor("#09101a")); p.drawRoundedRect(rect, 8, 8)
        if self.image is None:
            p.setPen(MUTED); p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No image yet"); return
        h, w = self.image.shape[:2]
        qimg = QImage(self.image.data, w, h, int(self.image.strides[0]), QImage.Format.Format_BGR888).copy()
        p.drawImage(rect, qimg)
        mask = self.selection_mask()
        if mask is not None and np.count_nonzero(mask) > 0:
            contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            p.setPen(QPen(ACCENT, 2, Qt.PenStyle.DashLine)); sx, sy = rect.width()/w, rect.height()/h
            for c in contours:
                x, y, cw, ch = cv2.boundingRect(c)
                p.drawRect(QRectF(rect.x()+x*sx, rect.y()+y*sy, cw*sx, ch*sy))


class ActivationGraph(QWidget):
    def __init__(self):
        super().__init__(); self.trace: Dict[str, object] = {"nodes": [], "edges": []}; self.setMinimumSize(600, 440)

    def set_trace(self, trace: Dict[str, object]) -> None:
        self.trace = trace; self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); p.fillRect(self.rect(), QColor("#090f18"))
        p.setPen(TEXT); p.setFont(QFont("Sans Serif", 13, 700)); p.drawText(18, 28, "CONCEPT FIRING / DISCOVERED STRUCTURE")
        nodes, edges = list(self.trace.get("nodes", [])), list(self.trace.get("edges", []))
        if not nodes:
            p.setPen(MUTED); p.drawText(QRectF(20, 60, self.width()-40, 80), Qt.AlignmentFlag.AlignCenter, "Train or generate an example to see sparse activations."); return
        kinds = ["word", "concept", "family", "feature"]; groups: Dict[str, List[dict]] = {k: [] for k in kinds}
        for node in nodes: groups.setdefault(str(node.get("kind", "concept")), []).append(node)
        x_map = {"word": .16, "concept": .34, "family": .54, "feature": .80}; positions: Dict[str, QPointF] = {}
        for kind in kinds:
            items = sorted(groups.get(kind, []), key=lambda n: float(n.get("firing", 0)), reverse=True)[:12 if kind == "feature" else 10]
            if not items: continue
            x = self.width()*x_map[kind]; p.setPen(QColor("#7186a5")); p.setFont(QFont("Sans Serif", 8, 700)); p.drawText(QRectF(x-60, 42, 120, 18), Qt.AlignmentFlag.AlignCenter, kind.upper())
            for i, node in enumerate(items): positions[str(node["id"])] = QPointF(x, 70+(self.height()-115)*(i+.5)/len(items))
        for edge in edges:
            a, b = positions.get(str(edge.get("src"))), positions.get(str(edge.get("dst")))
            if a is None or b is None: continue
            weight = float(np.clip(float(edge.get("weight", .3)), 0, 1)); col = QColor(84,210,255,int(50+150*weight)) if edge.get("kind") == "grounds_in" else QColor(194,151,255,int(50+150*weight))
            p.setPen(QPen(col, .8+2.2*weight)); p.drawLine(a, b)
        color_map = {"word": PURPLE, "concept": GOOD, "family": WARN, "feature": ACCENT}
        for node in nodes:
            pos = positions.get(str(node["id"]));
            if pos is None: continue
            firing = float(np.clip(float(node.get("firing", 0)), 0, 1)); r = 9+11*firing
            p.setBrush(QBrush(color_map.get(str(node.get("kind")), GOOD))); p.setPen(QPen(QColor("#eaf2ff"), 1)); p.drawEllipse(pos, r, r)
            p.setPen(TEXT); p.setFont(QFont("Sans Serif", 8, 600)); p.drawText(QRectF(pos.x()-70, pos.y()+r+2, 140, 28), Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop, str(node.get("label", ""))[:22])


class APCNV08Window(QMainWindow):
    def __init__(self, seed: int = 8):
        super().__init__(); self.setWindowTitle("APCN V0.8 — Interactive Grounded Concept Laboratory"); self.resize(1540, 920)
        self.session = TrainingSessionV08(seed=seed); self.current: Optional[StepResult] = None; self.auto_remaining = self.auto_total = 0
        self.auto_timer = QTimer(self); self.auto_timer.timeout.connect(self._auto_tick); self._build_ui(); self._show_preview(); self._log("READY", "APCN V0.8 initialized. No neural network/backpropagation is used in this trainer.")

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root); outer = QVBoxLayout(root); outer.setContentsMargins(10,10,10,10); outer.setSpacing(8)
        header = QHBoxLayout(); title = QLabel("APCN V0.8  •  INTERACTIVE GROUNDED CONCEPT LAB"); title.setFont(QFont("Sans Serif",15,800)); header.addWidget(title); header.addStretch(); self.status_label = QLabel("episodes: 0"); header.addWidget(self.status_label); outer.addLayout(header)
        controls = QGroupBox("Teacher + training controls"); grid = QGridLayout(controls)
        self.color_combo = QComboBox(); self.color_combo.addItem("random"); self.color_combo.addItems(self.session.teacher.color_words)
        self.shape_combo = QComboBox(); self.shape_combo.addItem("random"); self.shape_combo.addItems(self.session.teacher.shape_words)
        self.diff_slider = QSlider(Qt.Orientation.Horizontal); self.diff_slider.setRange(0,100); self.diff_slider.setValue(65); self.diff_label = QLabel("difficulty 0.65"); self.diff_slider.valueChanged.connect(lambda v: self.diff_label.setText(f"difficulty {v/100:.2f}"))
        self.generate_btn = QPushButton("Generate / test sample"); self.generate_btn.clicked.connect(self._show_preview)
        self.train1_btn = QPushButton("Train 1"); self.train1_btn.clicked.connect(lambda: self._train_steps(1))
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1,100000); self.batch_spin.setValue(250)
        self.auto_btn = QPushButton("Auto train"); self.auto_btn.setCheckable(True); self.auto_btn.clicked.connect(self._toggle_auto)
        self.eval_btn = QPushButton("Evaluate 200"); self.eval_btn.clicked.connect(self._evaluate)
        self.save_btn = QPushButton("Save memory"); self.save_btn.clicked.connect(self._save); self.load_btn = QPushButton("Load memory"); self.load_btn.clicked.connect(self._load_memory)
        grid.addWidget(QLabel("Color"),0,0); grid.addWidget(self.color_combo,0,1); grid.addWidget(QLabel("Shape"),0,2); grid.addWidget(self.shape_combo,0,3); grid.addWidget(self.diff_label,0,4); grid.addWidget(self.diff_slider,0,5,1,2); grid.addWidget(self.generate_btn,0,7)
        grid.addWidget(self.train1_btn,1,0); grid.addWidget(QLabel("Batch"),1,1); grid.addWidget(self.batch_spin,1,2); grid.addWidget(self.auto_btn,1,3); grid.addWidget(self.eval_btn,1,4); grid.addWidget(self.save_btn,1,5); grid.addWidget(self.load_btn,1,6); self.progress = QProgressBar(); grid.addWidget(self.progress,1,7); outer.addWidget(controls)
        split = QSplitter(Qt.Orientation.Horizontal); self.canvas = ImageCanvas(); self.graph = ActivationGraph(); split.addWidget(self.canvas); split.addWidget(self.graph); split.setSizes([760,760]); outer.addWidget(split,1)
        bottom = QSplitter(Qt.Orientation.Horizontal); left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0,0,0,0)
        self.teacher_sentence = QLabel("Teacher sentence: —"); self.teacher_sentence.setFont(QFont("Sans Serif",11,700)); self.prediction_label = QLabel("APCN prediction: —"); self.prediction_label.setStyleSheet("color:#61e294"); ll.addWidget(self.teacher_sentence); ll.addWidget(self.prediction_label)
        teachrow = QHBoxLayout(); self.manual_utterance = QLineEdit(); self.manual_utterance.setPlaceholderText("Teaching utterance for focused object, e.g. this is a yellow circle"); self.teach_btn = QPushButton("Teach current focus"); self.teach_btn.clicked.connect(self._teach_current); self.load_image_btn = QPushButton("Load real image"); self.load_image_btn.clicked.connect(self._load_real_image); teachrow.addWidget(self.manual_utterance,1); teachrow.addWidget(self.teach_btn); teachrow.addWidget(self.load_image_btn); ll.addLayout(teachrow)
        cmdrow = QHBoxLayout(); self.command = QLineEdit(); self.command.setPlaceholderText("train 100 | show yellow circle | inspect yellow | test 200 | save | load"); self.command.returnPressed.connect(self._run_command); run = QPushButton("Run"); run.clicked.connect(self._run_command); cmdrow.addWidget(self.command,1); cmdrow.addWidget(run); ll.addLayout(cmdrow); bottom.addWidget(left)
        self.tabs = QTabWidget(); self.answer = QPlainTextEdit(); self.log = QPlainTextEdit(); self.concepts = QPlainTextEdit(); self.families = QPlainTextEdit(); self.debug = QPlainTextEdit()
        for name, widget in [("Answer",self.answer),("Activity log",self.log),("Concepts",self.concepts),("Discovered families",self.families),("Debug",self.debug)]: widget.setReadOnly(True); self.tabs.addTab(widget,name)
        bottom.addWidget(self.tabs); bottom.setSizes([760,760]); outer.addWidget(bottom)

    def _log(self, tag: str, msg: str): self.log.appendPlainText(f"[{tag}] {msg}")
    def _choice(self, combo: QComboBox) -> Optional[str]: return None if combo.currentText() == "random" else combo.currentText()

    def _show_preview(self):
        r = self.session.generate_preview(self._choice(self.color_combo), self._choice(self.shape_combo), self.diff_slider.value()/100.0); self._display_result(r); self._log("PREVIEW", f"Generated {r.episode.teacher_metadata['color']} {r.episode.teacher_metadata['shape']}.")

    def _train_steps(self, count: int):
        last = None
        for _ in range(max(1,count)): last = self.session.step()
        if last: self._display_result(last); self._log("TRAIN", f"Completed {count} episode(s); total={self.session.learner.episode_count}; phase={last.state.phase}.")

    def _toggle_auto(self, checked: bool):
        if checked:
            self.auto_total = self.batch_spin.value(); self.auto_remaining = self.auto_total; self.progress.setValue(0); self.auto_btn.setText("Pause"); self.auto_timer.start(1); self._log("AUTO", f"Started {self.auto_total} episodes.")
        else:
            self.auto_timer.stop(); self.auto_btn.setText("Auto train"); self._log("AUTO", f"Paused with {self.auto_remaining} remaining.")

    def _auto_tick(self):
        if self.auto_remaining <= 0:
            self.auto_timer.stop(); self.auto_btn.blockSignals(True); self.auto_btn.setChecked(False); self.auto_btn.blockSignals(False); self.auto_btn.setText("Auto train"); self.progress.setValue(100); self._log("AUTO", "Batch complete."); return
        chunk = min(4,self.auto_remaining); last = None
        for _ in range(chunk): last = self.session.step()
        self.auto_remaining -= chunk; done = self.auto_total-self.auto_remaining; self.progress.setValue(int(100*done/max(1,self.auto_total)))
        if last and (done%20 == 0 or self.auto_remaining == 0): self._display_result(last)

    def _display_result(self, r: StepResult):
        self.current = r; ep = r.episode; truth = f"{ep.teacher_metadata.get('color','?')} {ep.teacher_metadata.get('shape','?')}"; pred = f"{r.predicted_color or '?'} {r.predicted_shape or '?'}"
        self.canvas.allow_selection = False; self.canvas.set_scene(ep.image, ep.attention_mask, "TEACHER / CAMERA VIEW", f"phase={r.state.phase} • dashed box = joint attention")
        self.teacher_sentence.setText(f"Teacher sentence:  {ep.utterance}"); self.prediction_label.setText(f"APCN prediction:  {pred}   |   teacher truth: {truth}   |   {'✓' if pred == truth else 'learning…'}")
        trace = self.session.learner.activation_trace(r.features, ep.utterance); self.graph.set_trace(trace); self.status_label.setText(f"episodes: {self.session.learner.episode_count} • vocabulary: {len(self.session.learner.token_stats)}")
        self.answer.setPlainText(f"Sentence\n  {ep.utterance}\n\nAPCN prediction\n  color: {r.predicted_color} ({r.color_score:.3f})\n  shape: {r.predicted_shape} ({r.shape_score:.3f})\n\nphase: {r.state.phase}\ndifficulty: {r.state.difficulty:.2f}\ntotal learned episodes: {self.session.learner.episode_count}\n")
        self._refresh_memory_panels(); self.debug.setPlainText(json.dumps({"teacher_metadata_FOR_EVALUATION_ONLY":ep.teacher_metadata,"state":r.state.__dict__,"prediction":{"color":r.predicted_color,"color_score":r.color_score,"shape":r.predicted_shape,"shape_score":r.shape_score},"trace":trace}, indent=2))

    def _refresh_memory_panels(self):
        rows = [f"{t:<14} quality={self.session.learner.concept_quality(t):0.3f} support={s.count:<5d} role={self.session.learner.role_guess(t)}" for t,s in sorted(self.session.learner.token_stats.items(), key=lambda kv:self.session.learner.concept_quality(kv[0]), reverse=True)]
        self.concepts.setPlainText("\n".join(rows) if rows else "No learned tokens yet.")
        fams = self.session.learner.discover_families(); self.families.setPlainText("\n\n".join(f"{f['id']} similarity={f['mean_similarity']:.3f} quality={f['mean_quality']:.3f}\n  "+", ".join(f["members"]) for f in fams) if fams else "No stable families yet. Train more episodes.")

    def _teach_current(self):
        text = self.manual_utterance.text().strip()
        if not text: QMessageBox.information(self,"Teaching utterance required","Type what you want to teach first."); return
        if self.canvas.image is None: return
        mask = self.canvas.selection_mask()
        if mask is None: QMessageBox.warning(self,"No focus","Generate an example or draw a focus box."); return
        try: x = self.session.teach_current(text,self.canvas.image,mask)
        except Exception as exc: QMessageBox.critical(self,"Teaching failed",str(exc)); return
        self._log("HUMAN TEACH", f"{text!r}; total={self.session.learner.episode_count}."); self.teacher_sentence.setText(f"Human teaching sentence:  {text}"); self.graph.set_trace(self.session.learner.activation_trace(x,text)); self._refresh_memory_panels(); self.status_label.setText(f"episodes: {self.session.learner.episode_count} • vocabulary: {len(self.session.learner.token_stats)}")

    def _load_real_image(self):
        path, _ = QFileDialog.getOpenFileName(self,"Load camera/image frame","","Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path: return
        image = cv2.imread(path,cv2.IMREAD_COLOR)
        if image is None: QMessageBox.critical(self,"Image error","OpenCV could not read that image."); return
        self.canvas.allow_selection = True; self.canvas.set_scene(image,None,"REAL IMAGE TEACHING","Drag a rectangle around the object, type a sentence, then Teach current focus."); self.teacher_sentence.setText("Human teacher: draw a focus box and type what the object is."); self.prediction_label.setText("APCN prediction: waiting for focus / teaching"); self._log("IMAGE", f"Loaded {path}.")

    def _evaluate(self):
        self._log("EVAL","Running 200 held-out samples…"); QApplication.processEvents(); r = self.session.evaluate(200,.90); self.answer.setPlainText(f"HELD-OUT EVALUATION\n\nepisodes learned: {r.episodes}\ncolor accuracy: {r.color_accuracy:.1%}\nshape accuracy: {r.shape_accuracy:.1%}\njoint accuracy: {r.joint_accuracy:.1%}\n"); self.tabs.setCurrentWidget(self.answer); self._log("EVAL",f"color={r.color_accuracy:.1%}, shape={r.shape_accuracy:.1%}, joint={r.joint_accuracy:.1%}")

    def _save(self):
        path = self.session.save(); self._log("SAVE",f"Saved {path}."); self.answer.setPlainText(f"Saved APCN V0.8 memory to:\n{path}")

    def _load_memory(self):
        path, _ = QFileDialog.getOpenFileName(self,"Load APCN V0.8 memory",str(Path("outputs/v0_8/concept_memory_v0_8.json")),"JSON (*.json)")
        if not path: return
        try: self.session = TrainingSessionV08.load(path)
        except Exception as exc: QMessageBox.critical(self,"Load failed",str(exc)); return
        self._log("LOAD",f"Loaded {path}; episodes={self.session.learner.episode_count}."); self._refresh_memory_panels(); self._show_preview()

    def _run_command(self):
        text = self.command.text().strip(); self.command.clear()
        if not text: return
        self._log("COMMAND",text); parts = text.lower().split()
        try:
            if parts[0] == "train":
                n = int(parts[1]) if len(parts)>1 else 1
                if n>1000: self.batch_spin.setValue(n); self.auto_btn.click() if not self.auto_btn.isChecked() else None
                else: self._train_steps(n)
                return
            if parts[0] == "show":
                colors = [p for p in parts if p in self.session.teacher.color_words]; shapes = [p for p in parts if p in self.session.teacher.shape_words]; self._display_result(self.session.generate_preview(colors[0] if colors else None, shapes[0] if shapes else None, self.diff_slider.value()/100.0)); return
            if parts[0] == "inspect" and len(parts)>1:
                token = parts[1]; profile = self.session.learner.token_profile(token); profile["role"] = self.session.learner.role_guess(token); profile["diagnostic_signal_mass_HUMAN_ONLY"] = self.session.learner.diagnostic_group_mass(token); self.answer.setPlainText(json.dumps(profile,indent=2)); self.tabs.setCurrentWidget(self.answer); return
            if parts[0] in ("test","eval","evaluate"):
                n = int(parts[1]) if len(parts)>1 else 200; r = self.session.evaluate(n,.90); self.answer.setPlainText(json.dumps(r.to_dict(),indent=2)); self.tabs.setCurrentWidget(self.answer); return
            if parts[0] == "save": self._save(); return
            if parts[0] == "load": self._load_memory(); return
        except Exception as exc: self._log("ERROR",str(exc)); self.answer.setPlainText(f"Command failed: {exc}"); return
        self.answer.setPlainText("Supported commands:\n  train 100\n  show yellow circle\n  inspect yellow\n  test 200\n  save\n  load\n\nUse the teaching utterance field for free-form English teaching.")


def run_app(seed: int = 8) -> int:
    app = QApplication.instance() or QApplication([]); app.setStyle("Fusion"); app.setStyleSheet(APP_STYLE); win = APCNV08Window(seed=seed); win.showMaximized(); return app.exec()
