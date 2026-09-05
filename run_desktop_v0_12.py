from __future__ import annotations
import argparse


def main() -> int:
    p = argparse.ArgumentParser(description="APCN V0.12 self-organizing perception studio")
    p.add_argument("--seed", type=int, default=12)
    args = p.parse_args()
    try:
        from apcn_v12.ui import run_app
    except ImportError as exc:
        if "PyQt6" in str(exc):
            print("PyQt6 is not installed. Run ./install_linux.sh")
            return 2
        raise
    return run_app(args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
