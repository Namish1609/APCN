from __future__ import annotations

from pathlib import Path
import json

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QPlainTextEdit,
    QFileDialog, QMessageBox,
)


def install_english_exposure_panel(window) -> QWidget:
    """Install a corpus-exposure tab on an APCNV15Window instance."""
    root = QWidget(); outer = QVBoxLayout(root); outer.setContentsMargins(10,10,10,10)
    title = QLabel("English Exposure — Surface Familiarity, Not Fake Understanding")
    title.setFont(QFont("Sans Serif", 14, 800)); outer.addWidget(title)
    hint = QLabel(
        "Paste English text or load a UTF-8 .txt file. APCN stores bounded word/phrase/context statistics and discards the raw text. "
        "This makes vocabulary and constructions familiar, but it does NOT automatically claim semantic understanding. Meaning must still be grounded or explicitly taught."
    )
    hint.setWordWrap(True); outer.addWidget(hint)

    source = QPlainTextEdit(); source.setPlaceholderText("Paste English text here, or load a .txt file..."); outer.addWidget(source, 2)
    row = QHBoxLayout()
    load = QPushButton("Load .txt")
    ingest = QPushButton("Ingest Surface Exposure")
    coverage = QPushButton("Analyze Current Text")
    clear = QPushButton("Clear Text Box")
    for b in (load, ingest, coverage, clear): row.addWidget(b)
    outer.addLayout(row)

    status = QLabel("No English corpus exposure ingested yet."); status.setWordWrap(True); outer.addWidget(status)
    audit = QPlainTextEdit(); audit.setReadOnly(True); outer.addWidget(audit, 1)

    def refresh(extra=None):
        payload = window.cognitive.english_exposure_v15.summary(20)
        if extra is not None:
            payload = {"last_operation": extra, **payload}
        audit.setPlainText(json.dumps(payload, indent=2, default=str))

    def do_load():
        path, _ = QFileDialog.getOpenFileName(window, "Load English text", "", "Text files (*.txt);;All files (*)")
        if not path: return
        try:
            text = Path(path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            QMessageBox.critical(window, "Load failed", str(exc)); return
        source.setPlainText(text)
        status.setText(f"Loaded {len(text):,} characters into working text box. Not learned yet.")

    def do_ingest():
        text = source.toPlainText()
        if not text.strip():
            QMessageBox.information(window, "No text", "Paste or load English text first."); return
        try:
            result = window.cognitive.ingest_english_text(text)
            window.cognitive.save("outputs/v0_15")
            status.setText(
                f"Ingested {result['tokens_added']:,} tokens / {result['sentences_added']:,} sentences. "
                "Surface statistics updated; semantic learning=False; raw text retained=False."
            )
            refresh(result)
        except Exception as exc:
            QMessageBox.critical(window, "Ingest failed", str(exc))

    def do_coverage():
        text = source.toPlainText()
        if not text.strip():
            QMessageBox.information(window, "No text", "Paste or load English text first."); return
        result = window.cognitive.english_coverage(text)
        status.setText(
            f"surface familiar {result['surface_familiar']:.1%} • semantic-anchor token fraction {result['semantic_anchor']:.1%} • "
            f"unknown surface tokens {len(result['unknown'])}"
        )
        refresh({"coverage": result})

    load.clicked.connect(do_load)
    ingest.clicked.connect(do_ingest)
    coverage.clicked.connect(do_coverage)
    clear.clicked.connect(source.clear)
    window.tabs.addTab(root, "English Exposure")
    refresh()
    return root
