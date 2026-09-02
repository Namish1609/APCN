from __future__ import annotations
import argparse
import json
from apcn_v08.session import TrainingSessionV08


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("memory")
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--difficulty", type=float, default=0.92)
    args = p.parse_args()
    session = TrainingSessionV08.load(args.memory)
    report = session.evaluate(args.samples, args.difficulty)
    print(json.dumps(report.to_dict(), indent=2))
    print("\nConcept families:")
    print(json.dumps(session.learner.discover_families(), indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
