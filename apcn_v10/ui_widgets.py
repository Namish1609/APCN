from __future__ import annotations

from typing import Dict, List, Optional
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QBrush
from PyQt6.QtWidgets import QWidget

STYLE = """
QWidget { background:#0b1119; color:#edf4ff; font-size:12px; }
QFrame#card { background:#121b27; border:1px solid #27364a; border-radius:9px; }
QFrame#run { background:#152536; border:1px solid #2d7197; border-radius:9px; }
QFrame#pause { background:#302512; border:1px solid #9d742b; border-radius:9px; }
QFrame#done { background:#132d21; border:1px solid #378359; border-radius:9px; }
QPushButton { background:#1b2939; color:#fff; border:1px solid #344b67; border-radius:7px; padding:8px 12px; font-weight:700; }
QPushButton:hover { background:#24364b; }
QPushButton#primary { background:#155b7a; border-color:#2b8eb8; }
QLineEdit,QSpinBox,QPlainTextEdit,QTableWidget { background:#0e1621; color:#fff; border:1px solid #2e4057; border-radius:6px; padding:5px; }
QTabWidget::pane { border:1px solid #27364a; border-radius:8px; }
QTabBar::tab { background:#101923; color:#9eb0c7; padding:8px 14px; border:1px solid #27364a; }
QTabBar::tab:selected { background:#1b2939; color:#fff; }
QProgressBar { background:#0e1621; border:1px solid #2e4057; border-radius:5px; text-align:center; min-height:22px; }
QProgressBar::chunk { background:#2d94ba; }
QHeaderView::section { background:#182333; color:#dfe9f7; border:1px solid #2d3e54; padding:4px; font-weight:700; }
"""


class ImageCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.image: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.title = "Perception teacher"
        self.subtitle = ""
        self.setMinimumSize(590, 480)

    def set_scene(self, image: np.ndarray, mask: Optional[np.ndarray], title: str, subtitle: str = "") -> None:
        self.image = np.ascontiguousarray(image.copy())
        self.mask = None if mask is None else np.ascontiguousarray(mask.copy())
        self.title, self.subtitle = title, subtitle
        self.update()

    def _image_rect(self) -> QRectF:
        avail = QRectF(16, 55, max(10, self.width()-32), max(10, self.height()-72))
        if self.image is None:
            return avail
        h, w = self.image.shape[:2]
        scale = min(avail.width()/w, avail.height()/h)
        rw, rh = w*scale, h*scale
        return QRectF(avail.x()+(avail.width()-rw)/2, avail.y()+(avail.height()-rh)/2, rw, rh)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#101722"))
        p.setPen(QColor("#f3f7fd")); p.setFont(QFont("Sans Serif", 12, 700)); p.drawText(16, 24, self.title)
        p.setPen(QColor("#93a7c0")); p.setFont(QFont("Sans Serif", 9)); p.drawText(16, 44, self.subtitle[:120])
        rect = self._image_rect()
        p.setPen(QPen(QColor("#32455d"), 1)); p.setBrush(QColor("#070c12")); p.drawRoundedRect(rect, 8, 8)
        if self.image is None:
            p.setPen(QColor("#93a7c0")); p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Start perception training")
            return
        h, w = self.image.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        qimg = QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format.Format_RGB888).copy()
        p.drawImage(rect, qimg)
        if self.mask is not None and np.count_nonzero(self.mask):
            contours, _ = cv2.findContours((self.mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor("#56c7ff"), 2, Qt.PenStyle.DashLine))
            sx, sy = rect.width()/w, rect.height()/h
            for c in contours:
                x, y, cw, ch = cv2.boundingRect(c)
                p.drawRect(QRectF(rect.x()+x*sx, rect.y()+y*sy, cw*sx, ch*sy))


class LearningCurve(QWidget):
    def __init__(self):
        super().__init__()
        self.history: List[Dict[str, float]] = []
        self.setMinimumHeight(235)

    def set_history(self, history: List[Dict[str, float]]) -> None:
        self.history = list(history)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#0d151f"))
        p.setPen(QColor("#e9f2ff")); p.setFont(QFont("Sans Serif", 10, 700)); p.drawText(12, 19, "Generated-test learning curve")
        box = QRectF(48, 34, max(40, self.width()-65), max(60, self.height()-62))
        p.setPen(QPen(QColor("#40536b"), 1)); p.drawRect(box)
        for frac in (0.0, .25, .5, .75, 1.0):
            y = box.bottom() - frac*box.height()
            p.setPen(QPen(QColor("#263648"), 1)); p.drawLine(QPointF(box.left(),y), QPointF(box.right(),y))
            p.setPen(QColor("#8fa2b9")); p.drawText(8, int(y+4), f"{int(frac*100)}")
        if not self.history:
            p.setPen(QColor("#8fa2b9")); p.drawText(box, Qt.AlignmentFlag.AlignCenter, "Run generated tests at different training stages")
            return
        xmin = min(h["episodes"] for h in self.history); xmax = max(h["episodes"] for h in self.history)
        if xmax <= xmin: xmax = xmin + 1
        metrics = (("exact", QColor("#61e294")), ("intent", QColor("#54d2ff")), ("relation", QColor("#ffc857")), ("operator", QColor("#c297ff")))
        for key, color in metrics:
            points = []
            for h in self.history:
                x = box.left() + (h["episodes"]-xmin)/(xmax-xmin)*box.width()
                y = box.bottom() - float(h[key])*box.height()
                points.append(QPointF(x,y))
            p.setPen(QPen(color, 2))
            for a,b in zip(points, points[1:]): p.drawLine(a,b)
            p.setBrush(QBrush(color)); p.setPen(QPen(color,1))
            for q in points: p.drawEllipse(q, 3.5, 3.5)
        p.setFont(QFont("Sans Serif",8))
        x = 55
        for key,color in metrics:
            p.setPen(color); p.drawText(x, self.height()-8, key); x += 72
        p.setPen(QColor("#8fa2b9")); p.drawText(int(box.left()), int(box.bottom()+17), str(int(xmin))); p.drawText(int(box.right()-35), int(box.bottom()+17), str(int(xmax)))
