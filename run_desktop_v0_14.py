from __future__ import annotations

import argparse
import sys

from PyQt6.QtWidgets import QApplication

from apcn_v14.launch_window import APCNV14LaunchWindow


def main() -> int:
    p = argparse.ArgumentParser(description="APCN V0.14.1 language-first live cognition studio")
    p.add_argument("--seed", type=int, default=14)
    args = p.parse_args()
    app = QApplication(sys.argv)
    window = APCNV14LaunchWindow(seed=args.seed)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
