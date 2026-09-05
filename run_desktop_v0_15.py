from __future__ import annotations

import argparse
import sys

from PyQt6.QtWidgets import QApplication

from apcn_v15.ui import APCNV15Window


def main() -> int:
    parser = argparse.ArgumentParser(description="APCN V0.15 conversational language studio")
    parser.add_argument("--seed", type=int, default=15)
    args = parser.parse_args()
    app = QApplication(sys.argv)
    window = APCNV15Window(seed=args.seed)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
