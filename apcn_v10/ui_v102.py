from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPlainTextEdit,
    QProgressBar, QPushButton, QSpinBox, QSplitter, QTableWidget,
    QVBoxLayout, QWidget,
)

from .ui_widgets import ImageCanvas
from .ui_v101 import APCNV101Window
from .ui_v101_widgets import SpikingGraph


class APCNV102Window(APCNV101Window):
    """V0.10.2 UI hotfix.

    V0.10.1 appended activation widgets by searching for the last QFrame in
    each page.  At 1366x768 that could place the graph in a squeezed/hidden
    container.  V0.10.2 makes activation a first-class region of each
    training layout, so it is always visible and resizable.
    """

    def _install_training_spike_graphs(self) -> None:
        # APCNV101Window calls this while its base UI is being constructed.
        # V0.10.2 replaces the three training tabs explicitly after init, so
        # do not append widgets heuristically to the old pages.
        return

    def __init__(self, seed: int = 10):
        super().__init__(seed)
        self.setWindowTitle("APCN V0.10.2 — Automatic Learning Studio")

        # Preserve Ask APCN + Testing Ground from V0.10.1, but replace the
        # first three training pages with deterministic split layouts.
        for index in (2, 1, 0):
            self.tabs.removeTab(index)
        self.tabs.insertTab(0, self._perception_tab_v102(), "Perception")
        self.tabs.insertTab(1, self._language_tab_v102(), "Language")
        self.tabs.insertTab(2, self._definitions_tab_v102(), "Definitions")
        self._refresh_competence()
        self._refresh_definition_table()
        self._refresh_header()

    def _perception_tab_v102(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(6, 6, 6, 6)
        row.setSpacing(8)

        visual_split = QSplitter(Qt.Orientation.Vertical)
        self.p_canvas = ImageCanvas()
        self.p_canvas.setMinimumSize(520, 300)
        self.p_spikes = SpikingGraph("PERCEPTION FIRING / SPIKING — CURRENT SAMPLE")
        self.p_spikes.setMinimumHeight(185)
        visual_split.addWidget(self.p_canvas)
        visual_split.addWidget(self.p_spikes)
        visual_split.setStretchFactor(0, 3)
        visual_split.setStretchFactor(1, 2)
        visual_split.setSizes([410, 215])
        row.addWidget(visual_split, 3)

        side = self._card()
        side.setMaximumWidth(470)
        s = QVBoxLayout(side)
        s.setContentsMargins(10, 9, 10, 9)
        s.setSpacing(8)
        h = QLabel("Automatic shape + color grounding")
        h.setFont(QFont("Sans Serif", 12, 800))
        s.addWidget(h)
        info = QLabel(
            "Pixels/attention are learned online. The lower-left activation graph "
            "shows the word/concept/feature nodes currently active for the sample."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#9db0c7")
        s.addWidget(info)

        ctrl = QHBoxLayout()
        self.p_batch = QSpinBox()
        self.p_batch.setRange(50, 20000)
        self.p_batch.setValue(500)
        self.p_batch.setSingleStep(100)
        self.p_button = QPushButton("Start Auto Train")
        self.p_button.setObjectName("primary")
        self.p_button.clicked.connect(self._toggle_perception)
        ctrl.addWidget(QLabel("experiences"))
        ctrl.addWidget(self.p_batch)
        ctrl.addWidget(self.p_button, 1)
        s.addLayout(ctrl)

        self.p_progress = QProgressBar()
        self.p_progress.setRange(0, 1000)
        self.p_progress.setFormat("0.0%")
        s.addWidget(self.p_progress)
        self.p_status = QLabel("0 / 500 complete • 500 remaining • 0.0%\nphase: waiting")
        self.p_status.setWordWrap(True)
        self.p_status.setFont(QFont("Sans Serif", 10, 700))
        s.addWidget(self.p_status)
        self.p_current = QLabel("Current sample: —")
        self.p_current.setWordWrap(True)
        s.addWidget(self.p_current)
        self.p_hint = QLabel(
            "Pause/resume preserves the unfinished batch. The activation graph is "
            "diagnostic: APCN is not a spiking neural network."
        )
        self.p_hint.setWordWrap(True)
        self.p_hint.setStyleSheet("color:#8fa3bb")
        s.addWidget(self.p_hint)
        s.addStretch(1)
        row.addWidget(side, 2)
        return page

    def _language_tab_v102(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(7, 7, 7, 7)
        row.setSpacing(8)

        left = self._card()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(10, 9, 10, 9)
        ll.setSpacing(7)
        h = QLabel("Automatic grounded semantic language acquisition")
        h.setFont(QFont("Sans Serif", 12, 800))
        ll.addWidget(h)
        info = QLabel(
            "APCN chooses its own curriculum skill. The activation graph below the "
            "semantic programs shows sentence cues, learned concepts and operators firing."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#9db0c7")
        ll.addWidget(info)

        ctrl = QHBoxLayout()
        self.l_batch = QSpinBox()
        self.l_batch.setRange(100, 50000)
        self.l_batch.setValue(1200)
        self.l_batch.setSingleStep(200)
        self.l_button = QPushButton("Start Language Learning")
        self.l_button.setObjectName("primary")
        self.l_button.clicked.connect(self._toggle_language)
        ctrl.addWidget(QLabel("experiences"))
        ctrl.addWidget(self.l_batch)
        ctrl.addWidget(self.l_button, 1)
        ll.addLayout(ctrl)

        self.l_progress = QProgressBar()
        self.l_progress.setRange(0, 1000)
        self.l_progress.setFormat("0.0%")
        ll.addWidget(self.l_progress)
        self.l_status = QLabel("0 / 1200 complete • 1200 remaining • 0.0%\ncurrent skill: waiting")
        self.l_status.setFont(QFont("Sans Serif", 10, 700))
        self.l_status.setWordWrap(True)
        ll.addWidget(self.l_status)
        self.l_sentence = QLabel("Teacher sentence: —")
        self.l_sentence.setWordWrap(True)
        ll.addWidget(self.l_sentence)

        semantic_split = QSplitter(Qt.Orientation.Vertical)
        programs = QWidget()
        pl = QHBoxLayout(programs)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(6)
        self.l_expected = QPlainTextEdit()
        self.l_predicted = QPlainTextEdit()
        self.l_expected.setReadOnly(True)
        self.l_predicted.setReadOnly(True)
        pl.addWidget(self._labeled("World meaning", self.l_expected), 1)
        pl.addWidget(self._labeled("APCN before learning", self.l_predicted), 1)
        self.l_spikes = SpikingGraph("LANGUAGE FIRING / SPIKING — CURRENT SENTENCE")
        self.l_spikes.setMinimumHeight(185)
        semantic_split.addWidget(programs)
        semantic_split.addWidget(self.l_spikes)
        semantic_split.setStretchFactor(0, 3)
        semantic_split.setStretchFactor(1, 2)
        semantic_split.setSizes([290, 210])
        ll.addWidget(semantic_split, 1)
        row.addWidget(left, 3)

        right = self._card()
        right.setMaximumWidth(400)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(10, 9, 10, 9)
        ch = QLabel("Curriculum competence")
        ch.setFont(QFont("Sans Serif", 11, 800))
        rl.addWidget(ch)
        self.competence = QTableWidget()
        self.competence.setColumnCount(3)
        self.competence.setHorizontalHeaderLabels(["Skill", "Evidence", "Recent competence"])
        self.competence.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.competence.verticalHeader().setVisible(False)
        rl.addWidget(self.competence, 1)
        note = QLabel(
            "These are internal curriculum skills, not buttons you choose. Low competence "
            "causes more evidence to be generated automatically."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#8fa3bb")
        rl.addWidget(note)
        row.addWidget(right, 2)
        return page

    def _definitions_tab_v102(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        top = self._card()
        tl = QHBoxLayout(top)
        tl.setContentsMargins(10, 8, 10, 8)
        self.def_button = QPushButton("Learn science definition curriculum")
        self.def_button.setObjectName("primary")
        self.def_button.clicked.connect(self._learn_definition_curriculum)
        self.def_input = QLineEdit()
        self.def_input.setPlaceholderText("e.g. energy ratio is work divided by time")
        self.def_input.returnPressed.connect(self._teach_definition)
        teach = QPushButton("Teach definition")
        teach.clicked.connect(self._teach_definition)
        tl.addWidget(self.def_button)
        tl.addWidget(self.def_input, 1)
        tl.addWidget(teach)
        outer.addWidget(top)

        main_split = QSplitter(Qt.Orientation.Horizontal)
        table_card = self._card()
        tcl = QVBoxLayout(table_card)
        tcl.setContentsMargins(8, 8, 8, 8)
        tcl.addWidget(QLabel("Concept dependency memory"))
        self.def_table = QTableWidget()
        self.def_table.setColumnCount(5)
        self.def_table.setHorizontalHeaderLabels(
            ["Concept", "Kind", "Definition", "Dependencies", "Grounding audit"]
        )
        self.def_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.def_table.verticalHeader().setVisible(False)
        tcl.addWidget(self.def_table, 1)
        main_split.addWidget(table_card)

        right_split = QSplitter(Qt.Orientation.Vertical)
        audit = self._card()
        al = QVBoxLayout(audit)
        al.setContentsMargins(8, 8, 8, 8)
        ah = QLabel("Concept-from-concept status")
        ah.setFont(QFont("Sans Serif", 11, 800))
        al.addWidget(ah)
        self.def_status = QPlainTextEdit()
        self.def_status.setReadOnly(True)
        self.def_status.setPlainText(
            "Definitions create explicit dependency structures. Missing prerequisites "
            "remain unresolved instead of being treated as understood."
        )
        al.addWidget(self.def_status, 1)
        self.def_spikes = SpikingGraph("DEFINITION FIRING / SPIKING — ACTIVE DEPENDENCY PATH")
        self.def_spikes.setMinimumHeight(190)
        right_split.addWidget(audit)
        right_split.addWidget(self.def_spikes)
        right_split.setStretchFactor(0, 3)
        right_split.setStretchFactor(1, 2)
        right_split.setSizes([360, 220])
        main_split.addWidget(right_split)
        main_split.setStretchFactor(0, 3)
        main_split.setStretchFactor(1, 2)
        main_split.setSizes([780, 500])
        outer.addWidget(main_split, 1)
        return page


def run_app(seed: int = 10) -> int:
    import sys
    from PyQt6.QtWidgets import QApplication
    from .ui_widgets import STYLE

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = APCNV102Window(seed)
    win.show()
    return app.exec()
