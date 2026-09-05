from __future__ import annotations

from pathlib import Path
from typing import Optional
import time

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QLineEdit, QDoubleSpinBox, QCheckBox, QFileDialog,
    QMessageBox, QTableWidget, QTableWidgetItem, QPlainTextEdit, QSplitter,
    QGroupBox,
)

from apcn_v10.ui_widgets import STYLE
from apcn_v11.ui import APCNV11Window
from apcn_v12.ui import APCNV12Window, ConsolidationWorkerV12
from .session import CognitiveSessionV13


class ConsolidationWorkerV13(ConsolidationWorkerV12):
    """Same diagnostic protocol as V0.12 but isolated V0.13 persistence."""

    def run(self):
        try:
            self.progress.emit(5, "diagnosing perception")
            v0 = self.session.test_visual(self.visual_test, self.difficulty)
            self.progress.emit(18, "diagnosing language / reference")
            l0 = self.session.test_language(self.language_test)
            before = {
                "visual_joint": v0.joint_accuracy,
                "visual_shape": v0.shape_accuracy,
                "language_exact": l0.exact_accuracy,
                "language_intent": l0.intent_accuracy,
                "language_reference": l0.skill_accuracy.get("reference", 0.0),
            }
            planned = [p.__dict__ for p in self.session.prescriptions(16)]
            self.progress.emit(35, "targeted visual consolidation")
            vt = self.session.consolidate_visual(self.visual_train)
            self.progress.emit(58, "recent-weighted language correction")
            lt = self.session.consolidate_language(self.language_train)
            self.progress.emit(76, "retesting perception")
            v1 = self.session.test_visual(self.visual_test, self.difficulty)
            self.progress.emit(88, "retesting language / reference")
            l1 = self.session.test_language(self.language_test)
            after = {
                "visual_joint": v1.joint_accuracy,
                "visual_shape": v1.shape_accuracy,
                "language_exact": l1.exact_accuracy,
                "language_intent": l1.intent_accuracy,
                "language_reference": l1.skill_accuracy.get("reference", 0.0),
            }
            result = {"before": before, "after": after, "visual_training": vt,
                      "language_training": lt, "prescriptions": planned}
            self.session.consolidation_history.append(result)
            self.session.save("outputs/v0_13")
            self.progress.emit(100, "cycle complete")
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class APCNV13Window(APCNV12Window):
    def __init__(self, seed: int = 13):
        super().__init__(seed)
        self.seed = seed
        self.setWindowTitle("APCN V0.13 — Persistent World Memory Studio")

        out13 = Path("outputs/v0_13")
        out12 = Path("outputs/v0_12")
        if (out13 / "session_v0_13.json").exists():
            self.cognitive = CognitiveSessionV13.load_checkpoint(out13, seed=seed)
            note = "loaded V0.13 world checkpoint"
        elif (out12 / "session_v0_12.json").exists():
            self.cognitive = CognitiveSessionV13.from_v12_checkpoint(out12, seed=seed)
            self.cognitive.save(out13)
            note = "imported V0.12 cognition; new persistent instance memory"
        else:
            self.cognitive = CognitiveSessionV13(seed)
            note = "new V0.13 memory"

        self.perception = self.cognitive.visual
        self.query_engine = self.cognitive.query
        self.visual_history = self.cognitive.visual_test_history
        self.consolidation_history = self.cognitive.consolidation_history
        self.consolidation_worker = None
        self._world_image: Optional[np.ndarray] = None
        self._world_path = ""
        self._world_clock = time.time()

        # Camera frames are working state only. They are never written into the
        # APCN checkpoint; only compact descriptors/world beliefs are retained.
        self._camera = None
        self._camera_index = 0
        self._camera_timer = QTimer(self)
        self._camera_timer.setInterval(66)
        self._camera_timer.timeout.connect(self._camera_tick)

        self.c_migrate.setText("Re-import V0.12 Knowledge")
        self.p_hint.setText(
            "V0.13 keeps V0.12 self-organizing perception and adds bounded multi-view INSTANCE memory, "
            "temporal identity, VISIBLE/OCCLUDED/OUT_OF_VIEW/LOST beliefs and explicit human correction."
        )
        self.p_current.setText(f"V0.13: {note}")
        self.tabs.addTab(self._build_world_tab(), "World Memory")
        self._refresh_header(); self._refresh_representation_status(); self._refresh_world()

    def _build_world_tab(self) -> QWidget:
        root = QWidget(); outer = QVBoxLayout(root); outer.setContentsMargins(10,10,10,10)
        title = QLabel("Persistent Object / World Memory")
        title.setFont(QFont("Sans Serif", 14, 800)); outer.addWidget(title)
        hint = QLabel(
            "Load a real image or use a live desktop camera, define a normalized focus box, teach a persistent object name, then observe/match later frames. "
            "Long-term memory stores bounded appearance prototypes and world beliefs—not camera frames. Camera mode is for object/world-memory testing and does not implement biometric face identity.")
        hint.setWordWrap(True); outer.addWidget(hint)

        split = QSplitter(Qt.Orientation.Horizontal); outer.addWidget(split, 1)
        left = QWidget(); ll = QVBoxLayout(left); split.addWidget(left)
        right = QWidget(); rl = QVBoxLayout(right); split.addWidget(right)
        split.setSizes([520,720])

        self.w_preview = QLabel("Load an image/frame or start camera")
        self.w_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.w_preview.setMinimumHeight(260)
        self.w_preview.setStyleSheet("border:1px solid #445; background:#101318")
        ll.addWidget(self.w_preview, 1)

        box = QGroupBox("Observation controls"); grid = QGridLayout(box); ll.addWidget(box)
        self.w_load = QPushButton("Load Image / Frame"); self.w_load.clicked.connect(self._world_load)
        self.w_cam_start = QPushButton("Start Camera"); self.w_cam_start.clicked.connect(self._camera_start)
        self.w_cam_freeze = QPushButton("Freeze Frame"); self.w_cam_freeze.clicked.connect(self._camera_freeze)
        self.w_cam_stop = QPushButton("Stop Camera"); self.w_cam_stop.clicked.connect(self._camera_stop)
        grid.addWidget(self.w_load,0,0); grid.addWidget(self.w_cam_start,0,1)
        grid.addWidget(self.w_cam_freeze,0,2); grid.addWidget(self.w_cam_stop,0,3)
        self.w_cam_status = QLabel("camera: stopped")
        self.w_cam_status.setWordWrap(True); grid.addWidget(self.w_cam_status,1,0,1,4)

        self.w_name = QLineEdit(); self.w_name.setPlaceholderText("persistent object name, e.g. my_bottle")
        self.w_category = QLineEdit(); self.w_category.setPlaceholderText("optional category, e.g. bottle")
        grid.addWidget(QLabel("Name"),2,0); grid.addWidget(self.w_name,2,1,1,3)
        grid.addWidget(QLabel("Category"),3,0); grid.addWidget(self.w_category,3,1,1,3)

        self.w_box = []
        for col,(label,val) in enumerate((("x",.20),("y",.20),("w",.60),("h",.60))):
            sp = QDoubleSpinBox(); sp.setRange(0.0,1.0); sp.setDecimals(3); sp.setSingleStep(.02); sp.setValue(val)
            sp.valueChanged.connect(self._world_preview_refresh); self.w_box.append(sp)
            grid.addWidget(QLabel(label),4,col*2 if col < 2 else (col-2)*2)
            grid.addWidget(sp,4 if col < 2 else 5,(col*2+1) if col < 2 else ((col-2)*2+1))
        self.w_teach = QPushButton("Teach Named Instance"); self.w_teach.clicked.connect(self._world_teach)
        self.w_observe = QPushButton("Observe / Match (no learning label)"); self.w_observe.clicked.connect(self._world_observe)
        self.w_correct = QPushButton("Correct Current Observation To Name"); self.w_correct.clicked.connect(self._world_correct)
        grid.addWidget(self.w_teach,6,0,1,4); grid.addWidget(self.w_observe,7,0,1,4); grid.addWidget(self.w_correct,8,0,1,4)
        self.w_occluded = QCheckBox("missing frame is due to an occluder near current box")
        self.w_missing = QPushButton("No Detection This Frame"); self.w_missing.clicked.connect(self._world_missing)
        grid.addWidget(self.w_occluded,9,0,1,4); grid.addWidget(self.w_missing,10,0,1,4)

        query_row = QHBoxLayout(); self.w_query = QLineEdit(); self.w_query.setPlaceholderText("my_bottle")
        qb = QPushButton("Where is it?"); qb.clicked.connect(self._world_where)
        query_row.addWidget(self.w_query,1); query_row.addWidget(qb); ll.addLayout(query_row)
        self.w_answer = QLabel("Belief answer appears here."); self.w_answer.setWordWrap(True); ll.addWidget(self.w_answer)

        self.w_tracks = QTableWidget(0,7)
        self.w_tracks.setHorizontalHeaderLabels(["name","instance","state","confidence","center","views","corrections"])
        self.w_tracks.horizontalHeader().setStretchLastSection(True); rl.addWidget(QLabel("Persistent tracks")); rl.addWidget(self.w_tracks,2)
        self.w_events = QPlainTextEdit(); self.w_events.setReadOnly(True); rl.addWidget(QLabel("Recent world events")); rl.addWidget(self.w_events,2)
        self.w_memory = QPlainTextEdit(); self.w_memory.setReadOnly(True); rl.addWidget(QLabel("World-memory audit")); rl.addWidget(self.w_memory,1)
        save = QPushButton("Save V0.13 Checkpoint"); save.clicked.connect(lambda: self.cognitive.save("outputs/v0_13")); rl.addWidget(save)
        return root

    def _bbox(self):
        x,y,w,h = [s.value() for s in self.w_box]
        w = min(w,1.0-x); h=min(h,1.0-y)
        return (x,y,max(.01,w),max(.01,h))

    def _category(self):
        return tuple(x.strip().lower() for x in self.w_category.text().split(",") if x.strip())

    def _world_load(self):
        self._camera_stop(silent=True)
        path,_ = QFileDialog.getOpenFileName(self,"Load observation frame","","Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path: return
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.warning(self,"Image error","Could not load image."); return
        self._world_image = image; self._world_path = path; self._world_preview_refresh()

    def _camera_start(self):
        self._camera_stop(silent=True)
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            cap.release()
            self.w_cam_status.setText("camera: unavailable")
            QMessageBox.warning(
                self, "Camera unavailable",
                "Could not open camera device 0. On a local desktop, allow camera permission. "
                "A remote Xvfb/noVNC server usually has no physical camera unless one is forwarded."
            )
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        self._camera = cap
        self._camera_timer.start()
        self.w_cam_status.setText("camera: LIVE • current frame is RAM-only working state")

    def _camera_tick(self):
        if self._camera is None:
            return
        ok, frame = self._camera.read()
        if not ok or frame is None:
            self._camera_timer.stop()
            self.w_cam_status.setText("camera: read failed")
            return
        self._world_image = frame
        self._world_path = f"camera:{self._camera_index}"
        self._world_preview_refresh()

    def _camera_freeze(self):
        if self._camera is None:
            self.w_cam_status.setText("camera: not running")
            return
        self._camera_timer.stop()
        self.w_cam_status.setText("camera: FROZEN • use Teach/Observe on this frame, or Start Camera to resume")

    def _camera_stop(self, *, silent: bool = False):
        if hasattr(self, "_camera_timer"):
            self._camera_timer.stop()
        if self._camera is not None:
            try:
                self._camera.release()
            except Exception:
                pass
            self._camera = None
        if hasattr(self, "w_cam_status") and not silent:
            self.w_cam_status.setText("camera: stopped")

    def _world_preview_refresh(self):
        if self._world_image is None: return
        img = self._world_image.copy(); h,w = img.shape[:2]
        x,y,bw,bh = self._bbox(); p0=(int(x*w),int(y*h)); p1=(int((x+bw)*w),int((y+bh)*h))
        cv2.rectangle(img,p0,p1,(255,255,255),2)
        q = QImage(img.data,w,h,img.strides[0],QImage.Format.Format_BGR888).copy()
        pix = QPixmap.fromImage(q).scaled(self.w_preview.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation)
        self.w_preview.setPixmap(pix)

    def _timestamp(self):
        self._world_clock += 1.0; return self._world_clock

    def _require_image(self):
        if self._world_image is None:
            QMessageBox.information(self,"Load an image","Load an image/frame first."); return False
        return True

    def _world_teach(self):
        if not self._require_image(): return
        name = self.w_name.text().strip()
        if not name: QMessageBox.information(self,"Name required","Enter a persistent object name."); return
        try:
            row = self.cognitive.teach_named_instance(name,self._world_image,bbox=self._bbox(),category=self._category(),timestamp=self._timestamp())
            self.w_answer.setText(f"Taught {name} • instance {row['instance_id']} • compact views {row['views']}")
            self.cognitive.save("outputs/v0_13"); self._refresh_world()
        except Exception as exc: QMessageBox.critical(self,"Teaching failed",str(exc))

    def _world_observe(self):
        if not self._require_image(): return
        try:
            row = self.cognitive.observe_object(self._world_image,bbox=self._bbox(),category=self._category(),timestamp=self._timestamp())
            self.w_answer.setText(
                f"identity={row.get('name') or row.get('instance_id','?')} • state={row.get('identity_state',row.get('state'))} • "
                f"score={row.get('score',0):.3f} • created_new={row.get('created_new',False)}")
            self.cognitive.save("outputs/v0_13"); self._refresh_world()
        except Exception as exc: QMessageBox.critical(self,"Observation failed",str(exc))

    def _world_correct(self):
        name = self.w_name.text().strip()
        if not name: QMessageBox.information(self,"Name required","Enter the correct persistent name first."); return
        try:
            row = self.cognitive.correct_last_identity(name,timestamp=self._timestamp())
            self.w_answer.setText(f"Correction stored immediately: this observation → {row['name']}")
            self.cognitive.save("outputs/v0_13"); self._refresh_world()
        except Exception as exc: QMessageBox.critical(self,"Correction failed",str(exc))

    def _world_missing(self):
        occ = [self._bbox()] if self.w_occluded.isChecked() else []
        self.cognitive.observe_absence(timestamp=self._timestamp(),occluders=occ)
        self.cognitive.save("outputs/v0_13"); self._refresh_world()

    def _world_where(self):
        name = self.w_query.text().strip() or self.w_name.text().strip()
        if not name: return
        row = self.cognitive.where(name); self.w_answer.setText(str(row.get("answer",row)))

    def _refresh_world(self):
        if not hasattr(self,"w_tracks"): return
        rows = []
        for iid,t in self.cognitive.world.tracks.items():
            inst = self.cognitive.world.instances.instances.get(iid)
            rows.append((inst.name if inst else "",iid,t.state,f"{t.confidence:.2f}",f"{t.center[0]:.2f},{t.center[1]:.2f}",
                         inst.positive.observations if inst else 0, inst.correction_events if inst else 0))
        self.w_tracks.setRowCount(len(rows))
        for r,row in enumerate(rows):
            for c,val in enumerate(row): self.w_tracks.setItem(r,c,QTableWidgetItem(str(val)))
        events = self.cognitive.world.events[-80:]
        self.w_events.setPlainText("\n".join(f"f{e.frame_index:04d} {e.kind:<18} {e.instance_id} c={e.confidence:.2f} {e.detail}" for e in events))
        m = self.cognitive.world.memory_summary(); im=m["instance_memory"]
        self.w_memory.setPlainText(
            f"instances: {im['instances']} • named: {im['named_instances']}\n"
            f"positive prototypes: {im['positive_prototypes']} • negative prototypes: {im['negative_prototypes']}\n"
            f"max views / instance: {im['max_views_per_instance']}\n"
            f"tracks: {m['tracks']} • states: {m['states']}\n"
            f"events retained: {m['events_retained']} / {m['max_events_retained']}\n"
            f"raw image frames retained: {im['raw_frames_retained']}\nraw video frames retained: {m['raw_video_frames_retained']}"
        )

    def _start_consolidation(self):
        if self.p_timer.isActive() or self.l_timer.isActive():
            QMessageBox.information(self,"Pause training first","Pause training before consolidation."); return
        if self.consolidation_worker is not None and self.consolidation_worker.isRunning(): return
        self.c_run.setEnabled(False); self.c_migrate.setEnabled(False); self.c_progress.setValue(1); self.c_progress.setFormat("starting")
        worker = ConsolidationWorkerV13(self.cognitive,visual_test=self.c_visual_test.value(),language_test=self.c_language_test.value(),
                                        visual_train=self.c_visual_train.value(),language_train=self.c_language_train.value(),difficulty=.82)
        worker.progress.connect(self._consolidation_progress); worker.completed.connect(self._consolidation_done)
        worker.failed.connect(self._consolidation_failed); self.consolidation_worker=worker; worker.start()

    def _consolidation_done(self,result):
        APCNV11Window._consolidation_done(self,result)
        self._refresh_representation_status(); self._refresh_world(); self.cognitive.save("outputs/v0_13")

    def _save_perception(self):
        try: self.cognitive.save("outputs/v0_13")
        except Exception: pass

    def _migrate_v010(self):
        out12=Path("outputs/v0_12")
        if not (out12/"session_v0_12.json").exists():
            QMessageBox.warning(self,"No V0.12 checkpoint found","Expected outputs/v0_12/session_v0_12.json"); return
        self.cognitive=CognitiveSessionV13.from_v12_checkpoint(out12,seed=self.seed)
        self.perception=self.cognitive.visual; self.query_engine=self.cognitive.query
        self.visual_history=self.cognitive.visual_test_history; self.consolidation_history=self.cognitive.consolidation_history
        self.cognitive.save("outputs/v0_13"); self._refresh_header(); self._refresh_representation_status(); self._refresh_world()
        QMessageBox.information(self,"V0.12 imported","Perception, language, definitions, errors and discourse were imported. V0.13 instance/world memory starts clean.")

    def _refresh_header(self):
        if not hasattr(self,"memory_label"): return
        learner=self.perception.learner; ecount=len(getattr(self.cognitive.errors,"signatures",{}))
        wm=getattr(self.cognitive,"world",None); instances=len(wm.instances.instances) if wm else 0
        self.memory_label.setText(
            f"visual {learner.episode_count} • language {self.cognitive.language.learner.episode_count} • "
            f"definitions {self.cognitive.concepts.definition_count} • persistent instances {instances} • errors {ecount}")

    def closeEvent(self, event):
        self._camera_stop(silent=True)
        try:
            self.cognitive.save("outputs/v0_13")
        except Exception:
            pass
        super().closeEvent(event)


def run_app(seed: int = 13) -> int:
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = APCNV13Window(seed); win.show(); return app.exec()
