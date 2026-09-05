from __future__ import annotations

import json

from apcn_v15.benchmark import run_conversation_benchmark


if __name__ == "__main__":
    report = run_conversation_benchmark()
    print(json.dumps(report.to_dict(), indent=2))
    if report.act_accuracy < .85:
        raise SystemExit("V0.15 act accuracy below release gate")
    if report.required_content_accuracy < .80:
        raise SystemExit("V0.15 required-content accuracy below release gate")
    if report.unknown_honesty < 1.0:
        raise SystemExit("V0.15 unknown-honesty gate failed")
    if report.learned_memory_accuracy < 1.0:
        raise SystemExit("V0.15 explicit-learning gate failed")
    if report.followup_accuracy < .80:
        raise SystemExit("V0.15 follow-up dialogue gate failed")
    if report.visual_experiences_changed != 0:
        raise SystemExit("V0.15 modified visual training state")
