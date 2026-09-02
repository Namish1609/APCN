from __future__ import annotations
import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="APCN V0.8.1 guided grounded-concept laboratory")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument(
        "--classic",
        action="store_true",
        help="Open the original V0.8 advanced laboratory UI instead of the guided UI.",
    )
    args = parser.parse_args()
    try:
        if args.classic:
            from apcn_v08.ui import run_app
        else:
            from apcn_v08.ui_simple import run_app
    except ImportError as exc:
        if "PyQt6" in str(exc):
            print("PyQt6 is not installed. Run: python -m pip install -r requirements.txt")
            return 2
        raise
    return run_app(seed=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
