from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

import cv2
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QLineEdit, QDoubleSpinBox, QSpinBox, QCheckBox, QPlainTextEdit,
    QGroupBox, QMessageBox, QSplitter,
)

from apcn_v13.ui import APCNV13Window
from .session import CognitiveSessionV14


class APCNV14Window(APCNV13Window):
    """V0.14 studio.

    V0.14 deliberately keeps the V0.13 object/world tab intact and adds two
    focused laboratories: language-first construction learning and an opt-in
    local SELF-face verification experiment. Camera frames remain working state.
    """

    def __init__(self, seed: int = 14):
        super().__init__(seed)
        self.seed = seed
        self.setWindowTitle("APCN V0.14 — Language-First Live Cognition Studio")

        out14 = Path("outputs/v0_14")
        out13 = Path("outputs/v0_13")
        if (out14 / "session_v0_14.json").exists():
            self.cognitive = CognitiveSessionV14.load_checkpoint(out14, seed=seed)
            note = "loaded V0.14 checkpoint"
        elif (out13 / "session_v0_13.json").exists():
            self.cognitive = CognitiveSessionV14.from_v13_checkpoint(out13, seed=seed)
            self.cognitive.save(out14)
            note = "imported V0.13 cognition; added V0.14 language + local self-face memory"
        else:
            self.cognitive = CognitiveSessionV14(seed)
            note = "new V0.14 memory"

        # Rebind inherited UI actions to V0.14 cognition. V0.14 save() redirects
        # historical V0.13 output requests into outputs/v0_14, so inherited world
        # buttons cannot accidentally overwrite the previous release checkpoint.
        self.perception = self.cognitive.visual
        self.query_engine = self.cognitive.query
        self.visual_history = self.cognitive.visual_test_history
        self.consolidation_history = self.cognitive.consolidation_history

        self._face_frame_counter = 0
        self._face_bbox = (.28, .14, .44, .64)
        self._language_demo = None

        self.p_hint.setText(
            "V0.14 is language-first: ~80% of automatic learning budget goes to richer semantic constructions. "
            "V0.13 perception/world memory remains active as grounding. A separate opt-in local self-face lab uses no neural face embedding model."
        )
        self.p_current.setText(f"V0.14: {note}")
        self.tabs.addTab(self._build_language_first_tab(), "Language First")
        self.tabs.addTab(self._build_self_face_tab(), "Self Face Camera")
        self._refresh_language_v14()
        self._refresh_face_v14()
        self._refresh_header()

    # ------------------------------------------------------------------ language
    def _build_language_first_tab(self) -> QWidget:
        root = QWidget(); outer = QVBoxLayout(root); outer.setContentsMargins(10, 10, 10, 10)
        title = QLabel("Language-First Construction Learning")
        title.setFont(QFont("Sans Serif", 14, 800)); outer.addWidget(title)
        hint = QLabel(
            "V0.14 keeps language-independent semantic programs but learns more of the surface construction that maps a sentence onto them. "
            "Held-out tests are read-only. The 80/20 control means language gets most automatic learning experiences while perception still receives some grounded maintenance."
        )
        hint.setWordWrap(True); outer.addWidget(hint)

        controls = QGroupBox("Learning budget and diagnostics"); grid = QGridLayout(controls); outer.addWidget(controls)
        self.l_ratio = QDoubleSpinBox(); self.l_ratio.setRange(.50, .95); self.l_ratio.setSingleStep(.05); self.l_ratio.setDecimals(2); self.l_ratio.setValue(self.cognitive.language_budget_ratio)
        grid.addWidget(QLabel("Language budget ratio"), 0, 0); grid.addWidget(self.l_ratio, 0, 1)
        self.l_steps = QSpinBox(); self.l_steps.setRange(20, 5000); self.l_steps.setValue(400); self.l_steps.setSingleStep(100)
        grid.addWidget(QLabel("Batch experiences"), 0, 2); grid.addWidget(self.l_steps, 0, 3)
        b1 = QPushButton("Train Language Batch"); b1.clicked.connect(self._language_train)
        b2 = QPushButton("Mixed Priority Batch"); b2.clicked.connect(self._language_mixed)
        b3 = QPushButton("Held-Out Test (memory frozen)"); b3.clicked.connect(self._language_test)
        grid.addWidget(b1, 1, 0, 1, 2); grid.addWidget(b2, 1, 2); grid.addWidget(b3, 1, 3)

        split = QSplitter(Qt.Orientation.Horizontal); outer.addWidget(split, 1)
        left = QWidget(); ll = QVBoxLayout(left); split.addWidget(left)
        right = QWidget(); rl = QVBoxLayout(right); split.addWidget(right)
        split.setSizes([620, 650])

        self.l_status = QLabel("No V0.14 language batch run yet."); self.l_status.setWordWrap(True); ll.addWidget(self.l_status)
        self.l_report = QPlainTextEdit(); self.l_report.setReadOnly(True); ll.addWidget(self.l_report, 1)

        teach = QGroupBox("Explicit human paraphrase teaching"); tg = QGridLayout(teach); rl.addWidget(teach)
        self.l_demo_text = QLabel("Generate a grounded demonstration first."); self.l_demo_text.setWordWrap(True)
        self.l_paraphrase = QLineEdit(); self.l_paraphrase.setPlaceholderText("type another sentence that should mean the same thing")
        gen = QPushButton("Generate Grounded Meaning"); gen.clicked.connect(self._language_demo_generate)
        store = QPushButton("Teach My Paraphrase Once"); store.clicked.connect(self._language_demo_teach)
        tg.addWidget(gen, 0, 0); tg.addWidget(store, 0, 1)
        tg.addWidget(self.l_demo_text, 1, 0, 1, 2); tg.addWidget(self.l_paraphrase, 2, 0, 1, 2)

        self.l_patterns = QPlainTextEdit(); self.l_patterns.setReadOnly(True); rl.addWidget(QLabel("Learned program constructions")); rl.addWidget(self.l_patterns, 1)
        save = QPushButton("Save V0.14 Checkpoint"); save.clicked.connect(self._save_v14); rl.addWidget(save)
        return root

    def _language_train(self):
        try:
            self.cognitive.language_budget_ratio = float(self.l_ratio.value())
            row = self.cognitive.language_first_train(self.l_steps.value())
            self.l_status.setText(
                f"language +{row['experiences_added']} experiences • correct-before-learning {row['correct_before_learning_rate']:.1%}"
            )
            self._save_v14(silent=True); self._refresh_language_v14(last=row)
        except Exception as exc:
            QMessageBox.critical(self, "Language training failed", str(exc))

    def _language_mixed(self):
        try:
            self.cognitive.language_budget_ratio = float(self.l_ratio.value())
            row = self.cognitive.mixed_priority_train(self.l_steps.value())
            self.l_status.setText(
                f"mixed batch {row['total_steps']} • language ratio {row['language_ratio']:.0%} • visual +{row['visual_experiences_added']}"
            )
            self._save_v14(silent=True); self._refresh_language_v14(last=row)
        except Exception as exc:
            QMessageBox.critical(self, "Mixed training failed", str(exc))

    def _language_test(self):
        try:
            samples = max(60, min(600, self.l_steps.value()))
            row = self.cognitive.test_rich_language(samples)
            self.l_status.setText(
                f"HELD-OUT • exact {row['exact']:.1%} • intent {row['intent']:.1%} • relation {row['relation']:.1%} • operator {row['operator']:.1%} • memory frozen={row['memory_frozen']}"
            )
            self._save_v14(silent=True); self._refresh_language_v14(last=row)
        except Exception as exc:
            QMessageBox.critical(self, "Language test failed", str(exc))

    def _language_demo_generate(self):
        self._language_demo = self.cognitive.language.teacher.v14_simple(skill="user_paraphrase")
        ep = self._language_demo
        self.l_demo_text.setText(f"Teacher sentence: {ep.utterance}\nMeaning: {ep.program.pretty()}")
        self.l_paraphrase.clear()

    def _language_demo_teach(self):
        if self._language_demo is None:
            self._language_demo_generate()
        text = self.l_paraphrase.text().strip()
        if not text:
            QMessageBox.information(self, "Paraphrase required", "Type your paraphrase first."); return
        try:
            before = self.cognitive.language.learner.parse(text)
            self.cognitive.language.teach_user_paraphrase(text, self._language_demo.program)
            after = self.cognitive.language.learner.parse(text)
            self.l_status.setText(
                f"explicit paraphrase stored • before={before.pretty() if before else 'unresolved'} • after={after.pretty() if after else 'needs more evidence'}"
            )
            self._save_v14(silent=True); self._refresh_language_v14()
        except Exception as exc:
            QMessageBox.critical(self, "Paraphrase teaching failed", str(exc))

    def _refresh_language_v14(self, last=None):
        if not hasattr(self, "l_patterns"): return
        s = self.cognitive.language.learner.program_constructions.summary(16)
        self.l_patterns.setPlainText(json.dumps(s, indent=2))
        if last is not None and hasattr(self, "l_report"):
            self.l_report.setPlainText(json.dumps(last, indent=2, default=str))
        elif hasattr(self, "l_report"):
            self.l_report.setPlainText(json.dumps({
                "language_episodes": self.cognitive.language.learner.episode_count,
                "language_budget_ratio": self.cognitive.language_budget_ratio,
                "history_tail": self.cognitive.v14_language_history[-6:],
            }, indent=2, default=str))

    # ---------------------------------------------------------- local self-face lab
    def _build_self_face_tab(self) -> QWidget:
        root = QWidget(); outer = QVBoxLayout(root); outer.setContentsMargins(10, 10, 10, 10)
        title = QLabel("Opt-In Local Self-Face Verification")
        title.setFont(QFont("Sans Serif", 14, 800)); outer.addWidget(title)
        hint = QLabel(
            "This is a self-test, not public-person identification and not security-grade authentication. "
            "Identity uses bounded APCN appearance prototypes—no pretrained neural face embedding encoder. "
            "The optional OpenCV Haar cascade only LOCATES a face box. You can disable auto-locate and set the box manually. "
            "Raw camera frames and face crops are not saved in the checkpoint. Enroll 8–12 varied views for a useful test."
        )
        hint.setWordWrap(True); outer.addWidget(hint)

        split = QSplitter(Qt.Orientation.Horizontal); outer.addWidget(split, 1)
        left = QWidget(); ll = QVBoxLayout(left); split.addWidget(left)
        right = QWidget(); rl = QVBoxLayout(right); split.addWidget(right); split.setSizes([650, 600])

        self.f_preview = QLabel("Start the camera in this tab or load a frame in World Memory")
        self.f_preview.setAlignment(Qt.AlignmentFlag.AlignCenter); self.f_preview.setMinimumHeight(330)
        self.f_preview.setStyleSheet("border:1px solid #445; background:#101318")
        ll.addWidget(self.f_preview, 1)

        cam = QHBoxLayout();
        bs = QPushButton("Start Camera"); bs.clicked.connect(self._camera_start)
        bf = QPushButton("Freeze"); bf.clicked.connect(self._camera_freeze)
        bp = QPushButton("Stop"); bp.clicked.connect(self._camera_stop)
        ba = QPushButton("Auto Locate Face"); ba.clicked.connect(self._face_auto_once)
        for b in (bs, bf, bp, ba): cam.addWidget(b)
        ll.addLayout(cam)

        controls = QGroupBox("Self enrollment / verification"); g = QGridLayout(controls); rl.addWidget(controls)
        self.f_name = QLineEdit("me"); g.addWidget(QLabel("Local alias"), 0, 0); g.addWidget(self.f_name, 0, 1, 1, 3)
        self.f_auto = QCheckBox("Auto-update face box while camera is live"); self.f_auto.setChecked(True); g.addWidget(self.f_auto, 1, 0, 1, 4)
        self.f_box = []
        defaults = (("x", .28), ("y", .14), ("w", .44), ("h", .64))
        for i, (label, value) in enumerate(defaults):
            sp = QDoubleSpinBox(); sp.setRange(0.0, 1.0); sp.setDecimals(3); sp.setSingleStep(.02); sp.setValue(value)
            sp.valueChanged.connect(self._face_preview_refresh); self.f_box.append(sp)
            g.addWidget(QLabel(label), 2 + i//2, (i%2)*2); g.addWidget(sp, 2 + i//2, (i%2)*2+1)
        en = QPushButton("Enroll Current Face View"); en.clicked.connect(self._face_enroll)
        vr = QPushButton("Verify: Is This Me?"); vr.clicked.connect(self._face_verify)
        ng = QPushButton("Correction: This Is NOT Me"); ng.clicked.connect(self._face_negative)
        g.addWidget(en, 4, 0, 1, 4); g.addWidget(vr, 5, 0, 1, 4); g.addWidget(ng, 6, 0, 1, 4)

        self.f_status = QLabel("No self identity enrolled."); self.f_status.setWordWrap(True); rl.addWidget(self.f_status)
        self.f_memory = QPlainTextEdit(); self.f_memory.setReadOnly(True); rl.addWidget(self.f_memory, 1)
        save = QPushButton("Save V0.14 Checkpoint"); save.clicked.connect(self._save_v14); rl.addWidget(save)
        return root

    def _face_bbox_values(self):
        x, y, w, h = [float(s.value()) for s in self.f_box]
        return (x, y, max(.01, min(w, 1.0-x)), max(.01, min(h, 1.0-y)))

    def _face_set_bbox(self, bbox):
        if bbox is None: return
        for sp, v in zip(self.f_box, bbox):
            sp.blockSignals(True); sp.setValue(float(v)); sp.blockSignals(False)
        self._face_preview_refresh()

    def _face_auto_once(self):
        if self._world_image is None:
            QMessageBox.information(self, "Camera/frame required", "Start the camera or load a frame first."); return
        bbox = self.cognitive.face_auto_bbox(self._world_image)
        if bbox is None:
            self.f_status.setText("Face locator found no face. Use the manual x/y/w/h box or improve lighting/frontal pose.")
        else:
            self._face_set_bbox(bbox); self.f_status.setText("Face box located. Identity has NOT been learned unless you press Enroll.")

    def _face_preview_refresh(self):
        if not hasattr(self, "f_preview") or self._world_image is None: return
        img = self._world_image.copy(); H, W = img.shape[:2]
        x, y, bw, bh = self._face_bbox_values(); p0=(int(x*W),int(y*H)); p1=(int((x+bw)*W),int((y+bh)*H))
        cv2.rectangle(img, p0, p1, (255,255,255), 2)
        q = QImage(img.data, W, H, img.strides[0], QImage.Format.Format_BGR888).copy()
        pix = QPixmap.fromImage(q).scaled(self.f_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.f_preview.setPixmap(pix)

    def _camera_tick(self):
        super()._camera_tick()
        self._face_frame_counter += 1
        if self._world_image is not None and hasattr(self, "f_auto") and self.f_auto.isChecked() and self._face_frame_counter % 5 == 0:
            bbox = self.cognitive.face_auto_bbox(self._world_image)
            if bbox is not None: self._face_set_bbox(bbox)
        self._face_preview_refresh()

    def _camera_freeze(self):
        super()._camera_freeze(); self._face_preview_refresh()

    def _face_require_frame(self):
        if self._world_image is None:
            QMessageBox.information(self, "Camera/frame required", "Start/freeze the camera or load a frame in World Memory first."); return False
        return True

    def _face_enroll(self):
        if not self._face_require_frame(): return
        name = self.f_name.text().strip() or "me"
        try:
            row = self.cognitive.enroll_self_face(name, self._world_image, self._face_bbox_values())
            self.f_status.setText(f"ENROLLED {row['name']} • views={row['views']} • prototype modes={row['prototype_modes']}")
            self._save_v14(silent=True); self._refresh_face_v14()
        except Exception as exc:
            QMessageBox.critical(self, "Face enrollment failed", str(exc))

    def _face_verify(self):
        if not self._face_require_frame(): return
        try:
            row = self.cognitive.recognize_self_face(self._world_image, self._face_bbox_values())
            verdict = "MATCH" if row.get("match") else "NOT COMMITTED"
            self.f_status.setText(f"{verdict} • state={row.get('state')} • score={row.get('score',0):.3f} • alias={row.get('name')}")
            self._refresh_face_v14()
        except Exception as exc:
            QMessageBox.critical(self, "Face verification failed", str(exc))

    def _face_negative(self):
        if not self._face_require_frame(): return
        try:
            row = self.cognitive.mark_face_not_me(self._world_image, self._face_bbox_values())
            self.f_status.setText(f"negative correction stored • score {row['score_before']:.3f} → {row['score_after']:.3f}")
            self._save_v14(silent=True); self._refresh_face_v14()
        except Exception as exc:
            QMessageBox.critical(self, "Negative correction failed", str(exc))

    def _refresh_face_v14(self):
        if not hasattr(self, "f_memory"): return
        self.f_memory.setPlainText(json.dumps(self.cognitive.self_face.summary(), indent=2, default=str))
        self._face_preview_refresh()

    def _save_v14(self, *, silent: bool = False):
        try:
            self.cognitive.save("outputs/v0_14")
            if not silent and hasattr(self, "l_status"):
                self.l_status.setText("V0.14 checkpoint saved.")
        except Exception as exc:
            if not silent: QMessageBox.critical(self, "Save failed", str(exc))

    def closeEvent(self, event):
        try:
            self.cognitive.save("outputs/v0_14")
        except Exception:
            pass
        super().closeEvent(event)
