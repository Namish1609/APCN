from __future__ import annotations

import argparse

from apcn_v18.ui import run_app


def main() -> int:
    parser = argparse.ArgumentParser(description="APCN V0.18 Natural Semantic Conversation desktop")
    parser.add_argument("--seed", type=int, default=18)
    args = parser.parse_args()
    return run_app(args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
