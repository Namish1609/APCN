from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRectF, QPointF, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QPlainTextEdit,
)

from .session import TrainingSessionV08, StepResult


APP_STYLE = """
QWidget { background:#0d1117; color:#eaf1fb; font-size:13px; }
QFrame#card { background:#151b24; border:1px solid #2b3544; border-radius:10px; }
QFrame#modePreview { background:#13263a; border:1px solid #2f78a4; border-radius:9px; }
QFrame#modeLearn { background:#322615; border:1px solid #a87928; border-radius:9px; }
QFrame#modeDone { background:#153022; border:1px solid #39875d; border-radius:9px; }
QPushButton { background:#202a38; color:#ffffff; border:1px solid #3a4b62; border-radius:8px; padding:9px 14px; font-weight:700; }
QPushButton:hover { background:#29364a; }
QPushButton#primary { background:#155a78; border-color:#2b8eb8; }
QPushButton#learn { background:#6a4b12; border-color:#ad7c22; }
QPushButton#danger { background:#653133; border-color:#9b4f52; }
QLineEdit,QComboBox,QSpinBox,QPlainTextEdit { background:#0f151e; color:#ffffff; border:1px solid #344154; border-radius:7px; padding:7px; }
QTabWidget::pane { border:1px solid #2b3544; border-radius:8px; }
QTabBar::tab { background:#121923; color:#9eb0c7; padding:8px 15px; border:1px solid #2b3544; }
QTabBar::tab:selected { background:#1e2938; color:#ffffff; }
QProgressBar { background:#0f151e; border:1px solid #344154; border-radius:6px; text-align:center; color:#ffffff; min-height:18px; }
QProgressBar::chunk { background:#2e8eb5; border-radius:5px; }
"""


class SceneCanvas(QWidget):
    """Simple image viewer with an optional focus rectangle.

    Images in APCN are OpenCV BGR arrays. We explicitly convert BGR -> RGB before
    constructing QImage. This is more portable than relying on QImage BGR888 on
    virtual X11/VNC stacks and fixes the black/wrong-color rendering seen on some
    headless Linux desktops.
    """

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
        self.title = title
        self.subtitle = subtitle
        self._drag_start = None
        self._drag_end = None
        self.update()

    def _image_rect(self) -> QRectF:
        avail = QRectF(16, 54, max(10, self.width() - 32), max(10, self.height() - 70))
        if self.image is None:
            return avail
        h, w = self.image.shape[:2]
        scale = min(avail.width() / w, avail.height() / h)
        rw, rh = w * scale, h * scale
        return QRectF(
            avail.x() + (avail.width() - rw) / 2,
            avail.y() + (avail.height() - rh) / 2,
            rw,
            rh,
        )

    def _widget_to_image(self, point: QPointF) -> Optional[Tuple[int, int]]:
        if self.image is None:
            return None
        rect = self._image_rect()
        if not rect.contains(point):
            return None
        h, w = self.image.shape[:2]
        x = int(np.clip((point.x() - rect.x()) / max(1.0, rect.width()) * w, 0, w - 1))
        y = int(np.clip((point.y() - rect.y()) / max(1.0, rect.height()) * h, 0, h - 1))
        return x, y

    def selection_mask(self) -> Optional[np.ndarray]:
        if self.image is None:
            return None
        if self._drag_start is None or self._drag_end is None:
            return self.mask
        a = self._widget_to_image(self._drag_start)
        b = self._widget_to_image(self._drag_end)
        if a is None or b is None:
            return self.mask
        x1, x2 = sorted((a[0], b[0]))
        y1, y2 = sorted((a[1], b[1]))
        if x2 - x1 < 3 or y2 - y1 < 3:
            return self.mask
        out = np.zeros(self.image.shape[:2], dtype=np.uint8)
        out[y1:y2 + 1, x1:x2 + 1] = 255
        return out

    def mousePressEvent(self, event):
        if (
            self.allow_selection
            and event.button() == Qt.MouseButton.LeftButton
            and self.image is not None
            and self._image_rect().contains(event.position())
        ):
            self._drag_start = event.position()
            self._drag_end = event.position()
            self.update()

    def mouseMoveEvent(self, event):
        if self.allow_selection and self._drag_start is not None:
            self._drag_end = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.allow_selection and self._drag_start is not None:
            self._drag_end = event.position()
            self.selectionChanged.emit(self.selection_mask())
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#111720"))

        p.setPen(QColor("#f2f6fc"))
        p.setFont(QFont("Sans Serif", 12, 700))
        p.drawText(16, 24, self.title)
        p.setPen(QColor("#96a9c0"))
        p.setFont(QFont("Sans Serif", 9))
        p.drawText(16, 43, self.subtitle[:100])

        rect = self._image_rect()
        p.setPen(QPen(QColor("#344154"), 1))
        p.setBrush(QColor("#080c12"))
        p.drawRoundedRect(rect, 8, 8)

        if self.image is None:
            p.setPen(QColor("#96a9c0"))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Generate an example")
            return

        h, w = self.image.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        qimg = QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format.Format_RGB888).copy()
        p.drawImage(rect, qimg)

        mask = self.selection_mask()
        if mask is not None and np.count_nonzero(mask) > 0:
            contours, _ = cv2.findContours(
                (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            p.setPen(QPen(QColor("#56c7ff"), 2, Qt.PenStyle.DashLine))
            sx, sy = rect.width() / w, rect.height() / h
            for c in contours:
                x, y, cw, ch = cv2.boundingRect(c)
                p.drawRect(QRectF(rect.x() + x * sx, rect.y() + y * sy, cw * sx, ch * sy))


class APCNSimpleWindow(QMainWindow):
    def __init__(self, seed: int = 8):
        super().__init__()
        self.setWindowTitle("APCN V0.8.1 — Guided Training")
        self.resize(1366, 768)
        self.setMinimumSize(1100, 650)

        self.session = TrainingSessionV08(seed=seed)
        self.current: Optional[StepResult] = None
        self.auto_total = 0
        self.auto_remaining = 0
        self.auto_timer = QTimer(self)
        self.auto_timer.timeout.connect(self._auto_tick)
        self._last_mode = "preview"

        self._build_ui()
        self._new_example()

    def _card(self) -> QFrame:
        f = QFrame()
        f.setObjectName("card")
        return f

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("APCN V0.8.1  •  GUIDED GROUNDED-CONCEPT TRAINING")
        title.setFont(QFont("Sans Serif", 14, 800))
        header.addWidget(title)
        header.addStretch()
        self.counter = QLabel("learned episodes: 0")
        self.counter.setStyleSheet("color:#9eb0c7")
        header.addWidget(self.counter)
        outer.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_learn_tab(), "Learn")
        self.tabs.addTab(self._build_inspect_tab(), "Inspect")
        self.tabs.addTab(self._build_manual_tab(), "Real image / manual")
        outer.addWidget(self.tabs, 1)

    def _build_learn_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self.canvas = SceneCanvas(allow_selection=False)
        layout.addWidget(self.canvas, 3)

        side = QWidget()
        side.setMaximumWidth(470)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(8)

        self.mode_card = QFrame()
        self.mode_card.setObjectName("modePreview")
        ml = QVBoxLayout(self.mode_card)
        ml.setContentsMargins(12, 9, 12, 9)
        self.mode_title = QLabel("PREVIEW ONLY — NOT LEARNING")
        self.mode_title.setFont(QFont("Sans Serif", 12, 800))
        self.mode_detail = QLabel("This example is only being inspected. Episode count will not change.")
        self.mode_detail.setWordWrap(True)
        self.mode_detail.setStyleSheet("color:#b8c8da")
        ml.addWidget(self.mode_title)
        ml.addWidget(self.mode_detail)
        sl.addWidget(self.mode_card)

        settings = self._card()
        setl = QHBoxLayout(settings)
        setl.setContentsMargins(10, 8, 10, 8)
        self.color_combo = QComboBox()
        self.color_combo.addItem("random")
        self.color_combo.addItems(self.session.teacher.color_words)
        self.shape_combo = QComboBox()
        self.shape_combo.addItem("random")
        self.shape_combo.addItems(self.session.teacher.shape_words)
        setl.addWidget(QLabel("Color"))
        setl.addWidget(self.color_combo)
        setl.addWidget(QLabel("Shape"))
        setl.addWidget(self.shape_combo)
        sl.addWidget(settings)

        actions = self._card()
        al = QVBoxLayout(actions)
        al.setContentsMargins(10, 10, 10, 10)
        self.new_btn = QPushButton("1. New / Test Example")
        self.new_btn.setObjectName("primary")
        self.new_btn.clicked.connect(self._new_example)
        self.teach_btn = QPushButton("2. Teach This Visible Example")
        self.teach_btn.setObjectName("learn")
        self.teach_btn.clicked.connect(self._teach_visible)

        batch_row = QHBoxLayout()
        self.batch_count = QSpinBox()
        self.batch_count.setRange(10, 10000)
        self.batch_count.setSingleStep(50)
        self.batch_count.setValue(100)
        self.auto_btn = QPushButton("3. Auto Train")
        self.auto_btn.clicked.connect(self._toggle_auto)
        batch_row.addWidget(QLabel("episodes"))
        batch_row.addWidget(self.batch_count)
        batch_row.addWidget(self.auto_btn, 1)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        al.addWidget(self.new_btn)
        al.addWidget(self.teach_btn)
        al.addLayout(batch_row)
        al.addWidget(self.progress)
        sl.addWidget(actions)

        result = self._card()
        rl = QVBoxLayout(result)
        rl.setContentsMargins(12, 10, 12, 10)
        heading = QLabel("What is happening now")
        heading.setFont(QFont("Sans Serif", 11, 800))
        self.teacher_line = QLabel("Teacher says: —")
        self.teacher_line.setWordWrap(True)
        self.truth_line = QLabel("Teacher truth: —")
        self.pred_line = QLabel("APCN predicts: —")
        self.pred_line.setFont(QFont("Sans Serif", 11, 700))
        self.result_line = QLabel("—")
        self.result_line.setWordWrap(True)
        rl.addWidget(heading)
        rl.addWidget(self.teacher_line)
        rl.addWidget(self.truth_line)
        rl.addWidget(self.pred_line)
        rl.addWidget(self.result_line)
        sl.addWidget(result)

        tutorial = self._card()
        tl = QVBoxLayout(tutorial)
        tl.setContentsMargins(12, 9, 12, 9)
        th = QLabel("Training tutorial")
        th.setFont(QFont("Sans Serif", 11, 800))
        body = QLabel(
            "① Click New/Test Example. APCN predicts, but does NOT learn.\n"
            "② Click Teach This Visible Example. That exact image + sentence is learned once.\n"
            "③ Use Auto Train for many generated lessons. The screen samples progress; every episode is learned.\n"
            "④ Click New/Test Example again to check generalization without changing memory."
        )
        body.setWordWrap(True)
        body.setStyleSheet("color:#b8c8da; line-height:1.3")
        tl.addWidget(th)
        tl.addWidget(body)
        sl.addWidget(tutorial)
        sl.addStretch()

        layout.addWidget(side, 2)
        return page

    def _build_inspect_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        left = self._card()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("Learned words — quality, support, inferred role"))
        self.concepts = QPlainTextEdit()
        self.concepts.setReadOnly(True)
        ll.addWidget(self.concepts, 1)

        right = self._card()
        rl = QVBoxLayout(right)
        rl.addWidget(QLabel("Discovered unlabeled word families"))
        self.families = QPlainTextEdit()
        self.families.setReadOnly(True)
        rl.addWidget(self.families, 1)

        controls = QHBoxLayout()
        save = QPushButton("Save memory")
        save.clicked.connect(self._save)
        load = QPushButton("Load memory")
        load.clicked.connect(self._load)
        evaluate = QPushButton("Evaluate 200")
        evaluate.clicked.connect(self._evaluate)
        controls.addWidget(save)
        controls.addWidget(load)
        controls.addWidget(evaluate)
        rl.addLayout(controls)
        self.eval_output = QLabel("Evaluation has not been run yet.")
        self.eval_output.setWordWrap(True)
        self.eval_output.setStyleSheet("color:#b8c8da")
        rl.addWidget(self.eval_output)

        layout.addWidget(left, 1)
        layout.addWidget(right, 1)
        return page

    def _build_manual_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self.manual_canvas = SceneCanvas(allow_selection=True)
        layout.addWidget(self.manual_canvas, 3)

        side = self._card()
        side.setMaximumWidth(430)
        sl = QVBoxLayout(side)
        title = QLabel("Teach from a real image")
        title.setFont(QFont("Sans Serif", 12, 800))
        instructions = QLabel(
            "1. Load an image.\n"
            "2. Drag a box around one object.\n"
            "3. Type a sentence such as ‘this is a yellow ball’.\n"
            "4. Click Teach focused object."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color:#b8c8da")
        load = QPushButton("Load image")
        load.clicked.connect(self._load_real_image)
        self.manual_sentence = QLineEdit()
        self.manual_sentence.setPlaceholderText("this is a yellow ball")
        teach = QPushButton("Teach focused object")
        teach.setObjectName("learn")
        teach.clicked.connect(self._teach_manual)
        self.manual_status = QLabel("No real image loaded.")
        self.manual_status.setWordWrap(True)
        self.manual_status.setStyleSheet("color:#b8c8da")
        sl.addWidget(title)
        sl.addWidget(instructions)
        sl.addWidget(load)
        sl.addWidget(self.manual_sentence)
        sl.addWidget(teach)
        sl.addWidget(self.manual_status)
        sl.addStretch()
        layout.addWidget(side, 2)
        return page

    def _choice(self, combo: QComboBox) -> Optional[str]:
        return None if combo.currentText() == "random" else combo.currentText()

    def _set_mode(self, mode: str, detail: str) -> None:
        self._last_mode = mode
        if mode == "learning":
            self.mode_card.setObjectName("modeLearn")
            self.mode_title.setText("LEARNING — MEMORY IS CHANGING")
        elif mode == "done":
            self.mode_card.setObjectName("modeDone")
            self.mode_title.setText("TRAINING COMPLETE")
        else:
            self.mode_card.setObjectName("modePreview")
            self.mode_title.setText("PREVIEW / TEST — NOT LEARNING")
        self.mode_detail.setText(detail)
        self.mode_card.style().unpolish(self.mode_card)
        self.mode_card.style().polish(self.mode_card)

    def _new_example(self) -> None:
        self.current = self.session.generate_preview(
            self._choice(self.color_combo),
            self._choice(self.shape_combo),
            difficulty=0.20,
            add_distractors=False,
        )
        self._set_mode(
            "preview",
            "APCN is predicting this example only. Episode count is unchanged until you click Teach.",
        )
        self._show_result(self.current)

    def _teach_visible(self) -> None:
        if self.current is None:
            self._new_example()
            return
        ep = self.current.episode
        before = self.session.learner.episode_count
        x = self.session.teach_current(ep.utterance, ep.image, ep.attention_mask)
        pc, pcs = self.session.learner.best_of(self.session.teacher.color_words, x)
        ps, pss = self.session.learner.best_of(self.session.teacher.shape_words, x)
        self.current.features = x
        self.current.predicted_color = pc
        self.current.color_score = pcs
        self.current.predicted_shape = ps
        self.current.shape_score = pss
        after = self.session.learner.episode_count
        self._set_mode(
            "learning",
            f"The visible example was learned exactly once. Episode count changed {before} → {after}.",
        )
        self._show_result(self.current)
        self._refresh_inspect()

    def _toggle_auto(self) -> None:
        if self.auto_timer.isActive():
            self.auto_timer.stop()
            self.auto_btn.setText("3. Auto Train")
            self._set_mode("preview", f"Training paused with {self.auto_remaining} episodes remaining.")
            return
        self.auto_total = self.batch_count.value()
        self.auto_remaining = self.auto_total
        self.progress.setValue(0)
        self.auto_btn.setText("Stop Training")
        self._set_mode(
            "learning",
            f"Automatic teacher is generating and learning {self.auto_total} grounded lessons.",
        )
        self.auto_timer.start(1)

    def _auto_tick(self) -> None:
        if self.auto_remaining <= 0:
            self.auto_timer.stop()
            self.auto_btn.setText("3. Auto Train")
            self.progress.setValue(100)
            path = self.session.save()
            self._set_mode("done", f"Batch finished and memory was saved to {path}.")
            self._refresh_inspect()
            return

        chunk = min(3, self.auto_remaining)
        last: Optional[StepResult] = None
        for _ in range(chunk):
            last = self.session.step()
        self.auto_remaining -= chunk
        done = self.auto_total - self.auto_remaining
        self.progress.setValue(int(100 * done / max(1, self.auto_total)))

        if last is not None and (done % 15 == 0 or self.auto_remaining == 0):
            self.current = last
            self._show_result(last)
            self._set_mode(
                "learning",
                f"Automatic training: {done}/{self.auto_total} lessons learned. The screen shows a sampled lesson, not every frame.",
            )

    def _show_result(self, r: StepResult) -> None:
        ep = r.episode
        color = str(ep.teacher_metadata.get("color", "?"))
        shape = str(ep.teacher_metadata.get("shape", "?"))
        truth = f"{color} {shape}"
        pred = f"{r.predicted_color or '?'} {r.predicted_shape or '?'}"
        color_ok = r.predicted_color == color
        shape_ok = r.predicted_shape == shape

        self.canvas.set_scene(
            ep.image,
            ep.attention_mask,
            f"Visible example: {truth}",
            "Dashed box = the object the teacher is talking about",
        )
        self.teacher_line.setText(f"Teacher says:  {ep.utterance}")
        self.truth_line.setText(f"Teacher truth:  {truth}")
        self.pred_line.setText(f"APCN predicts:  {pred}")

        if color_ok and shape_ok:
            self.result_line.setText("✓ color correct   ✓ shape correct")
            self.result_line.setStyleSheet("color:#62d99b")
        else:
            parts = ["✓ color" if color_ok else "✗ color", "✓ shape" if shape_ok else "✗ shape"]
            self.result_line.setText("   ".join(parts) + "   — wrong predictions are expected while learning")
            self.result_line.setStyleSheet("color:#f0c36a")

        self.counter.setText(
            f"learned episodes: {self.session.learner.episode_count}   •   vocabulary: {len(self.session.learner.token_stats)}"
        )
        self._refresh_inspect()

    def _refresh_inspect(self) -> None:
        rows = []
        for token, stats in sorted(
            self.session.learner.token_stats.items(),
            key=lambda kv: self.session.learner.concept_quality(kv[0]),
            reverse=True,
        ):
            rows.append(
                f"{token:<14} quality={self.session.learner.concept_quality(token):0.3f}  "
                f"support={stats.count:<5d}  role={self.session.learner.role_guess(token)}"
            )
        self.concepts.setPlainText("\n".join(rows) if rows else "No learned words yet.")

        fams = self.session.learner.discover_families()
        if fams:
            text = []
            for fam in fams:
                text.append(
                    f"{fam['id']}\n"
                    f"similarity={fam['mean_similarity']:.3f}  quality={fam['mean_quality']:.3f}\n"
                    f"members: {', '.join(fam['members'])}"
                )
            self.families.setPlainText("\n\n".join(text))
        else:
            self.families.setPlainText("No stable word families yet. Train more examples.")

    def _save(self) -> None:
        path = self.session.save()
        self.eval_output.setText(f"Memory saved to {path}")

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load APCN memory",
            str(Path("outputs/v0_8/concept_memory_v0_8.json")),
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            self.session = TrainingSessionV08.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self.counter.setText(f"learned episodes: {self.session.learner.episode_count}")
        self._refresh_inspect()
        self._new_example()
        self.eval_output.setText(f"Loaded {path}")

    def _evaluate(self) -> None:
        self.eval_output.setText("Running 200 held-out examples…")
        QApplication.processEvents()
        r = self.session.evaluate(200, 0.90)
        self.eval_output.setText(
            f"Held-out evaluation — color {r.color_accuracy:.1%}, "
            f"shape {r.shape_accuracy:.1%}, joint {r.joint_accuracy:.1%}. "
            "Evaluation does not teach the model."
        )

    def _load_real_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load real image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not path:
            return
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.critical(self, "Image error", "OpenCV could not read that image.")
            return
        self.manual_canvas.set_scene(
            image,
            None,
            "Real image",
            "Drag a rectangle around one object before teaching",
        )
        self.manual_status.setText("Image loaded. Draw a focus box, then type the teaching sentence.")

    def _teach_manual(self) -> None:
        text = self.manual_sentence.text().strip()
        if not text:
            QMessageBox.information(self, "Sentence required", "Type a teaching sentence first.")
            return
        if self.manual_canvas.image is None:
            QMessageBox.information(self, "Image required", "Load an image first.")
            return
        mask = self.manual_canvas.selection_mask()
        if mask is None or np.count_nonzero(mask) == 0:
            QMessageBox.information(self, "Focus required", "Drag a box around the object first.")
            return
        before = self.session.learner.episode_count
        self.session.teach_current(text, self.manual_canvas.image, mask)
        after = self.session.learner.episode_count
        self.manual_status.setText(f"Learned once. Episode count {before} → {after}.")
        self.counter.setText(f"learned episodes: {after}")
        self._refresh_inspect()


def run_app(seed: int = 8) -> int:
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    win = APCNSimpleWindow(seed=seed)
    win.show()
    return app.exec()
