from __future__ import annotations

import argparse
import sys

from PyQt6.QtWidgets import QApplication

from apcn_v16.ui import APCNV16Window


def main() -> int:
    parser = argparse.ArgumentParser(description="APCN V0.16 bidirectional semantic language studio")
    parser.add_argument("--seed", type=int, default=16)
    args = parser.parse_args()
    app = QApplication(sys.argv)
    window = APCNV16Window(seed=args.seed)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
