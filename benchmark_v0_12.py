from __future__ import annotations

import argparse
import json

from apcn_v12.benchmark import run_paired_benchmark


def main() -> int:
    p = argparse.ArgumentParser(description="Paired APCN V0.11 vs V0.12 perception benchmark")
    p.add_argument("--train", type=int, default=1800)
    p.add_argument("--test", type=int, default=600)
    p.add_argument("--difficulty", type=float, default=.86)
    p.add_argument("--seed", type=int, default=12012)
    args = p.parse_args()
    report = run_paired_benchmark(args.train, args.test, args.difficulty, args.seed)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
