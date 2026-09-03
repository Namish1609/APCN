from __future__ import annotations
import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="APCN V0.11 self-consolidating learning studio")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    try:
        from apcn_v11.ui import run_app
    except ImportError as exc:
        if "PyQt6" in str(exc):
            print("PyQt6 is not installed. Run ./install_linux.sh")
            return 2
        raise
    return run_app(args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
