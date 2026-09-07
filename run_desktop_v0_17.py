from __future__ import annotations

import argparse
import sys

from PyQt6.QtWidgets import QApplication

from apcn_v17.ui import APCNV17Window


def main() -> int:
    parser = argparse.ArgumentParser(description="APCN V0.17 structured discourse semantics studio")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    app = QApplication(sys.argv)
    window = APCNV17Window(seed=args.seed)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
