from __future__ import annotations

import json

from apcn_v16.benchmark import run_bidirectional_benchmark


if __name__ == "__main__":
    report = run_bidirectional_benchmark()
    print(json.dumps(report.to_dict(), indent=2))
    if report.request_accuracy < .95:
        raise SystemExit("V0.16 semantic request/response gate failed")
    if report.generation_min_variants < 5:
        raise SystemExit("V0.16 generation diversity gate failed")
    if report.roundtrip_exact < .95:
        raise SystemExit("V0.16 semantic roundtrip gate failed")
    if report.unknown_honesty < 1.0:
        raise SystemExit("V0.16 unknown-honesty gate failed")
    if report.explicit_learning_transfer < 1.0:
        raise SystemExit("V0.16 explicit-learning transfer gate failed")
    if report.content_firewall < 1.0:
        raise SystemExit("V0.16 generation content-firewall gate failed")
    if report.visual_experiences_changed != 0:
        raise SystemExit("V0.16 modified visual training state")
    if report.recombination_accuracy < .75:
        raise SystemExit("V0.16 learned construction recombination gate failed")
    # dev_parse_accuracy intentionally remains diagnostic only. The original DEV
    # includes genuinely unseen lexical material and must not be phrase-patched.
