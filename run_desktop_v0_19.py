from __future__ import annotations

import argparse
import sys

from PyQt6.QtWidgets import QApplication

from apcn_v19.ui import APCNV19Window


def main() -> int:
    parser = argparse.ArgumentParser(description="APCN V0.19 online English + arithmetic desktop")
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    app = QApplication(sys.argv)
    window = APCNV19Window(seed=args.seed)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
