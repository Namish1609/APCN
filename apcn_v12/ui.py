from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QLabel, QMessageBox

from apcn_v11.ui import APCNV11Window, ConsolidationWorker
from .session import CognitiveSessionV12


class ConsolidationWorkerV12(ConsolidationWorker):
    """V0.11 consolidation protocol with V0.12 checkpoint isolation."""

    def run(self):
        try:
            self.progress.emit(5, "diagnosing V0.12 visual confusions")
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

            self.progress.emit(35, "balanced bootstrap / targeted visual contrast learning")
            vt = self.session.consolidate_visual(self.visual_train)
            self.progress.emit(58, "recent-weighted language correction learning")
            lt = self.session.consolidate_language(self.language_train)

            self.progress.emit(76, "retesting V0.12 perception")
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
                self.session.visual.learner,
                self.session.visual.teacher.shape_words,
                limit=30,
            ))
            objective = self.session.consolidation.objective(
                error_rate=1.0 - 0.5 * (v1.joint_accuracy + l1.exact_accuracy),
                active_edges=int(graph.get("edges", 0)),
                ambiguous_pairs=ambiguous,
            )
            result = {
                "before": before,
                "after": after,
                "visual_training": vt,
                "language_training": lt,
                "prescriptions": planned,
                "diagnostic_objective": objective,
            }
            self.session.consolidation_history.append(result)
            self.session.save("outputs/v0_12")
            self.progress.emit(100, "cycle complete")
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class APCNV12Window(APCNV11Window):
    """V0.12 keeps the V0.11 studio layout and swaps in the new learning core."""

    def __init__(self, seed: int = 12):
        # Base construction creates the proven V0.11/V0.10.2 widgets. We then
        # replace the active cognitive session; no additional cluttered tab is
        # added for representation learning.
        super().__init__(seed)
        self.seed = seed
        self.setWindowTitle("APCN V0.12 — Self-Organizing Perception Studio")

        out12 = Path("outputs/v0_12")
        out11 = Path("outputs/v0_11")
        if (out12 / "session_v0_12.json").exists():
            self.cognitive = CognitiveSessionV12.load_checkpoint(out12, seed=seed)
            migration_note = "loaded V0.12 checkpoint"
        elif (out11 / "session_v0_11.json").exists():
            self.cognitive = CognitiveSessionV12.from_v11_checkpoint(
                out11, seed=seed, representation_bootstrap=240)
            self.cognitive.save(out12)
            migration_note = "imported compatible V0.11 knowledge; new visual space recalibrating"
        else:
            self.cognitive = CognitiveSessionV12(seed)
            migration_note = "new V0.12 memory"

        self.perception = self.cognitive.visual
        self.query_engine = self.cognitive.query
        self.visual_history = self.cognitive.visual_test_history
        self.consolidation_history = self.cognitive.consolidation_history
        self.consolidation_worker = None

        self.c_migrate.setText("Re-import V0.11 Knowledge")
        self.p_hint.setText(
            "V0.12 learns a bounded unlabeled local-patch codebook from focused pixels. "
            "Shape classification no longer depends on Hu moments, fill ratio or aspect ratio; "
            "generic continuous photometric statistics preserve color information."
        )
        self.v12_rep_status = QLabel()
        self.v12_rep_status.setWordWrap(True)
        self.v12_rep_status.setFont(QFont("Sans Serif", 9, 700))
        self.v12_rep_status.setStyleSheet("color:#8fd3ff")
        parent = self.p_hint.parentWidget()
        if parent is not None and parent.layout() is not None:
            idx = parent.layout().indexOf(self.p_hint)
            parent.layout().insertWidget(max(0, idx + 1), self.v12_rep_status)

        self.p_current.setText(f"V0.12 representation: {migration_note}")
        self._refresh_representation_status()
        self._refresh_header()
        self._refresh_competence()
        self._refresh_definition_table()
        self._refresh_consolidation()

    def _refresh_representation_status(self):
        if not hasattr(self, "v12_rep_status"):
            return
        learner = self.perception.learner
        s = learner.patch_sensor.memory_summary()
        coverage = self.cognitive.visual_coverage()
        missing = len(coverage["missing_colors"]) + len(coverage["missing_shapes"])
        self.v12_rep_status.setText(
            f"representation • {s['codewords']}/{s['max_codewords']} patch codewords • "
            f"{s['patch_updates']} local updates • {learner.sensor.dim} anonymous dimensions\n"
            f"V0.12 visual evidence {learner.episode_count} • legacy V0.11 evidence "
            f"{learner.legacy_visual_episode_count} • missing grounded classes {missing}"
        )

    @staticmethod
    def _format_audit(a):
        base = APCNV11Window._format_audit(a)
        rep = a.get("v012_representation")
        coverage = a.get("v012_visual_coverage")
        correction = a.get("v012_recent_language_corrections", {})
        if not rep:
            return base
        sensor = rep["sensor"]
        strongest = correction.get("strongest", [])
        correction_line = "none yet"
        if strongest:
            top = strongest[0]
            correction_line = (
                f"{top['intent']} via '{top['pattern']}' "
                f"(confidence {top['confidence']:.2f}, recent support {top['recent_support']:.1f})"
            )
        return base + (
            "\n\nV0.12 REPRESENTATION\n"
            f"feature dimensions: {sensor['feature_dim']}\n"
            f"learned patch codewords: {sensor['codewords']} / {sensor['max_codewords']}\n"
            f"local patch updates: {sensor['patch_updates']}\n"
            f"raw local patches retained: {sensor['raw_patches_retained']}\n"
            f"legacy V0.11 visual episodes (not replayed): {rep['legacy_visual_episodes_not_replayed']}\n"
            f"new V0.12 concept episodes: {rep['concept_episodes_v012']}\n"
            f"visual factor coverage complete: {bool(coverage and coverage['complete'])}\n\n"
            "V0.12 RECENT LANGUAGE CORRECTION\n"
            f"recent construction observations: {correction.get('observations', 0)}\n"
            f"bounded recent patterns: {correction.get('patterns', 0)} / {correction.get('max_patterns', 4096)}\n"
            f"strongest recent correction: {correction_line}\n\n"
            "V0.11 visual statistics remain provenance only; they are never mixed into the new feature space."
        )

    def _start_consolidation(self):
        if self.p_timer.isActive() or self.l_timer.isActive():
            QMessageBox.information(
                self, "Pause training first",
                "Pause Perception/Language training before consolidation so one memory writer is active at a time.")
            return
        if self.consolidation_worker is not None and self.consolidation_worker.isRunning():
            return
        self.c_run.setEnabled(False)
        self.c_migrate.setEnabled(False)
        self.c_progress.setValue(1)
        self.c_progress.setFormat("starting")
        worker = ConsolidationWorkerV12(
            self.cognitive,
            visual_test=self.c_visual_test.value(),
            language_test=self.c_language_test.value(),
            visual_train=self.c_visual_train.value(),
            language_train=self.c_language_train.value(),
            difficulty=.82,
        )
        worker.progress.connect(self._consolidation_progress)
        worker.completed.connect(self._consolidation_done)
        worker.failed.connect(self._consolidation_failed)
        self.consolidation_worker = worker
        worker.start()

    def _migrate_v010(self):
        # The inherited button is intentionally repurposed: V0.12's immediate
        # predecessor is V0.11, whose error/discourse memory should be retained.
        if self.p_timer.isActive() or self.l_timer.isActive():
            QMessageBox.information(
                self, "Pause training first",
                "Pause training before replacing active memory.")
            return
        out11 = Path("outputs/v0_11")
        if not (out11 / "session_v0_11.json").exists():
            QMessageBox.warning(
                self, "No V0.11 checkpoint found",
                "Expected outputs/v0_11/session_v0_11.json")
            return
        self.cognitive = CognitiveSessionV12.from_v11_checkpoint(
            out11, seed=self.seed, representation_bootstrap=240)
        self.perception = self.cognitive.visual
        self.query_engine = self.cognitive.query
        self.visual_history = self.cognitive.visual_test_history
        self.consolidation_history = self.cognitive.consolidation_history
        self.cognitive.save("outputs/v0_12")
        self._refresh_representation_status()
        self._refresh_header()
        self._refresh_competence()
        self._refresh_definition_table()
        self._refresh_consolidation()
        QMessageBox.information(
            self, "V0.11 knowledge imported",
            "Language, definitions, discourse and error memory were imported.\n\n"
            "V0.11 visual statistics were not copied into the incompatible V0.12 feature space. "
            "Their episode count and error signatures are retained, and a label-free patch-codebook bootstrap was run.\n\n"
            "Run one Consolidation cycle to automatically ground the new visual representation and apply recent-weighted language corrections."
        )

    def _perception_tick(self):
        super()._perception_tick()
        self._refresh_representation_status()

    def _consolidation_done(self, result):
        super()._consolidation_done(result)
        self._refresh_representation_status()
        self.cognitive.save("outputs/v0_12")

    def _save_perception(self):
        try:
            self.cognitive.save("outputs/v0_12")
        except Exception:
            pass

    def _refresh_header(self):
        if not hasattr(self, "memory_label"):
            return
        learner = self.perception.learner
        errors = getattr(getattr(self, "cognitive", None), "errors", None)
        ecount = len(errors.signatures) if errors is not None else 0
        legacy = getattr(learner, "legacy_visual_episode_count", 0)
        self.memory_label.setText(
            f"visual V0.12 {learner.episode_count} (+{legacy} legacy) • "
            f"language {self.cognitive.language.learner.episode_count} • "
            f"definitions {self.cognitive.concepts.definition_count} • errors {ecount}"
        )


def run_app(seed: int = 12) -> int:
    import sys
    from PyQt6.QtWidgets import QApplication
    from apcn_v10.ui_widgets import STYLE
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = APCNV12Window(seed)
    win.show()
    return app.exec()
