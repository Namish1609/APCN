from __future__ import annotations

import argparse
import sys

from PyQt6.QtWidgets import QApplication

from apcn_v14.ui import APCNV14Window


def main() -> int:
    p = argparse.ArgumentParser(description="APCN V0.14 language-first live cognition studio")
    p.add_argument("--seed", type=int, default=14)
    args = p.parse_args()
    app = QApplication(sys.argv)
    window = APCNV14Window(seed=args.seed)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
