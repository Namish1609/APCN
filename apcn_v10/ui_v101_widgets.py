from __future__ import annotations

from typing import Dict, List
import math
from PyQt6.QtCore import Qt, QPointF, QRectF, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QBrush
from PyQt6.QtWidgets import QWidget


class SpikingGraph(QWidget):
    """Animated sparse APCN activation graph; not a biological/neural-network claim."""
    def __init__(self, title: str = "APCN FIRING / SPIKING"):
        super().__init__()
        self.title = title
        self.trace: Dict[str, object] = {"nodes": [], "edges": []}
        self.phase = 0.0
        self.timer = QTimer(self); self.timer.timeout.connect(self._pulse); self.timer.start(110)
        self.setMinimumHeight(190)

    def set_trace(self, trace: Dict[str, object]) -> None:
        self.trace = trace or {"nodes": [], "edges": []}
        self.update()

    def _pulse(self):
        self.phase += 0.5
        self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#0b121c"))
        p.setPen(QColor("#e9f2ff")); p.setFont(QFont("Sans Serif", 9, 700))
        p.drawText(10, 18, self.title)
        p.setPen(QColor("#6f849d")); p.setFont(QFont("Sans Serif", 7))
        p.drawText(10, 33, "sparse concept activation • visualization only, not neural-network neurons")
        nodes = list(self.trace.get("nodes", [])); edges = list(self.trace.get("edges", []))
        if not nodes:
            p.setPen(QColor("#8499b2")); p.drawText(QRectF(10,40,self.width()-20,self.height()-45), Qt.AlignmentFlag.AlignCenter, "No activation yet")
            return
        kinds_order = ["word", "concept", "semantic", "operator", "feature", "family"]
        grouped: Dict[str, List[dict]] = {k: [] for k in kinds_order}
        for n in nodes:
            grouped.setdefault(str(n.get("kind", "concept")), []).append(n)
        present = [k for k in kinds_order if grouped.get(k)]
        if not present:
            present = ["concept"]
        xmap = {k: (i+1)/(len(present)+1) for i,k in enumerate(present)}
        pos: Dict[str, QPointF] = {}
        for kind in present:
            arr = sorted(grouped.get(kind, []), key=lambda n: float(n.get("firing",0)), reverse=True)[:9]
            for i,n in enumerate(arr):
                y = 48 + (self.height()-65)*(i+.5)/max(1,len(arr))
                pos[str(n.get("id"))] = QPointF(self.width()*xmap[kind], y)
        for e in edges:
            a=pos.get(str(e.get("src"))); b=pos.get(str(e.get("dst")))
            if a is None or b is None: continue
            w=max(0.0,min(1.0,float(e.get("weight",.2))))
            p.setPen(QPen(QColor(82,158,215,int(40+145*w)), .8+1.5*w)); p.drawLine(a,b)
        colors = {
            "word":QColor("#c49cff"), "concept":QColor("#65dfa1"),
            "semantic":QColor("#55c9ff"), "operator":QColor("#ffc857"),
            "feature":QColor("#ff8fab"), "family":QColor("#f6bd60")
        }
        for n in nodes:
            q=pos.get(str(n.get("id")))
            if q is None: continue
            fire=max(0.0,min(1.0,float(n.get("firing",0))))
            pulse=1.0 + .10*math.sin(self.phase + (hash(str(n.get("id"))) % 9))
            r=(4.8+7.5*fire)*pulse
            kind=str(n.get("kind","concept"))
            p.setBrush(QBrush(colors.get(kind,QColor("#65dfa1")))); p.setPen(QPen(QColor("#dce9f8"),1)); p.drawEllipse(q,r,r)
            p.setPen(QColor("#dce7f5")); p.setFont(QFont("Sans Serif",7))
            p.drawText(QRectF(q.x()-45,q.y()+r+1,90,17),Qt.AlignmentFlag.AlignHCenter,str(n.get("label",""))[:16])


class VisualLearningCurve(QWidget):
    def __init__(self):
        super().__init__(); self.history: List[Dict[str,float]]=[]; self.setMinimumHeight(220)

    def set_history(self, history: List[Dict[str,float]]) -> None:
        self.history=list(history); self.update()

    def paintEvent(self, event):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); p.fillRect(self.rect(),QColor("#0d151f"))
        p.setPen(QColor("#e9f2ff")); p.setFont(QFont("Sans Serif",10,700)); p.drawText(12,19,"Visual test learning curve")
        box=QRectF(48,34,max(40,self.width()-65),max(60,self.height()-62)); p.setPen(QPen(QColor("#40536b"),1)); p.drawRect(box)
        for frac in (0,.25,.5,.75,1):
            y=box.bottom()-frac*box.height(); p.setPen(QPen(QColor("#263648"),1)); p.drawLine(QPointF(box.left(),y),QPointF(box.right(),y)); p.setPen(QColor("#8fa2b9")); p.drawText(8,int(y+4),str(int(frac*100)))
        if not self.history:
            p.setPen(QColor("#8fa2b9")); p.drawText(box,Qt.AlignmentFlag.AlignCenter,"Run perception tests at different training stages"); return
        xmin=min(h["episodes"] for h in self.history); xmax=max(h["episodes"] for h in self.history); xmax=xmax if xmax>xmin else xmin+1
        metrics=(("color",QColor("#54d2ff")),("shape",QColor("#ffc857")),("joint",QColor("#61e294")))
        for key,color in metrics:
            pts=[]
            for h in self.history:
                x=box.left()+(h["episodes"]-xmin)/(xmax-xmin)*box.width(); y=box.bottom()-float(h[key])*box.height(); pts.append(QPointF(x,y))
            p.setPen(QPen(color,2))
            for a,b in zip(pts,pts[1:]): p.drawLine(a,b)
            p.setBrush(QBrush(color)); p.setPen(QPen(color,1))
            for q in pts: p.drawEllipse(q,3.5,3.5)
        p.setFont(QFont("Sans Serif",8)); x=55
        for key,color in metrics: p.setPen(color); p.drawText(x,self.height()-8,key); x+=76
        p.setPen(QColor("#8fa2b9")); p.drawText(int(box.left()),int(box.bottom()+17),str(int(xmin))); p.drawText(int(box.right()-35),int(box.bottom()+17),str(int(xmax)))
