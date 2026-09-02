from __future__ import annotations
import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="APCN V0.8 interactive grounded-concept laboratory")
    parser.add_argument("--seed", type=int, default=8)
    args = parser.parse_args()
    try:
        from apcn_v08.ui import run_app
    except ImportError as exc:
        if "PyQt6" in str(exc):
            print("PyQt6 is not installed. Run: python -m pip install -r requirements.txt")
            return 2
        raise
    return run_app(seed=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
