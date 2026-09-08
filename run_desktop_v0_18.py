from __future__ import annotations

import argparse
import sys

from PyQt6.QtWidgets import QApplication

from apcn_v18.ui import APCNV18Window


def main() -> int:
    parser = argparse.ArgumentParser(description="APCN V0.18 Natural Semantic Conversation desktop")
    parser.add_argument("--seed", type=int, default=18)
    args = parser.parse_args()
    app = QApplication(sys.argv)
    window = APCNV18Window(seed=args.seed)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
