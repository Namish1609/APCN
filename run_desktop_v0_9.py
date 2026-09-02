from __future__ import annotations
import argparse


def main() -> int:
    p = argparse.ArgumentParser(description="APCN V0.9 grounded semantic language studio")
    p.add_argument("--seed", type=int, default=9)
    a = p.parse_args()
    try:
        from apcn_v09.ui import run_app
    except ImportError as exc:
        if "PyQt6" in str(exc):
            print("PyQt6 is not installed. Run ./install_linux.sh first.")
            return 2
        raise
    return run_app(seed=a.seed)


if __name__ == "__main__":
    raise SystemExit(main())
