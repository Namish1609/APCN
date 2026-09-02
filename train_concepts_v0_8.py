from __future__ import annotations
import argparse
import json
from apcn_v08.session import TrainingSessionV08


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2400)
    p.add_argument("--eval-samples", type=int, default=400)
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--output-dir", type=str, default="outputs/v0_8")
    args = p.parse_args()
    session = TrainingSessionV08.load(args.resume, seed=args.seed) if args.resume else TrainingSessionV08(seed=args.seed)
    for i in range(args.episodes):
        step = session.step()
        if (i + 1) % 250 == 0:
            l = session.learner
            print(f"[{i+1:5d}/{args.episodes}] total={l.episode_count:5d} phase={step.state.phase:<22} yellow={l.concept_quality('yellow'):.3f} circle={l.concept_quality('circle'):.3f}")
    report = session.evaluate(samples=args.eval_samples, difficulty=0.90)
    mem = session.save(args.output_dir)
    print(json.dumps(report.to_dict(), indent=2))
    print("\nDiscovered families:")
    print(json.dumps(session.learner.discover_families(), indent=2))
    print(f"\nSaved: {mem}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
