from __future__ import annotations
import argparse, json
from pathlib import Path
from apcn_v09.session import SemanticSessionV09


def main() -> int:
    p = argparse.ArgumentParser(description="Train APCN V0.9 grounded semantic language learner")
    p.add_argument("--episodes", type=int, default=2400)
    p.add_argument("--test-samples", type=int, default=500)
    p.add_argument("--output", default="outputs/v0_9")
    a = p.parse_args()
    s = SemanticSessionV09()
    s.train(a.episodes)
    rep = s.test(a.test_samples)
    hard = s.test(a.test_samples, held_out_templates=True, seed=91919)
    mem = s.save(a.output)
    report = {"train_episodes": s.learner.episode_count, "standard": rep.to_dict(), "held_out_templates": hard.to_dict(), "memory": str(mem)}
    out = Path(a.output) / "language_report_v0_9.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"episodes": s.learner.episode_count, "exact": rep.exact_accuracy, "intent": rep.intent_accuracy, "relation": rep.relation_accuracy, "operator": rep.operator_accuracy, "held_out_exact": hard.exact_accuracy, "memory": str(mem)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
