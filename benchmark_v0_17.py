from __future__ import annotations

import json

from apcn_v17.benchmark import run_structured_benchmark


if __name__ == "__main__":
    report = run_structured_benchmark()
    print(json.dumps(report.to_dict(), indent=2))
    if report.operator_parse_accuracy < .95:
        raise SystemExit("V0.17 structured operator parse gate failed")
    if report.nested_roundtrip_exact < .95:
        raise SystemExit("V0.17 nested semantic roundtrip gate failed")
    if report.generation_min_variants < 2:
        raise SystemExit("V0.17 generation diversity gate failed")
    if report.discourse_reference_accuracy < 1.0:
        raise SystemExit("V0.17 discourse-reference gate failed")
    if report.universal_inference_accuracy < 1.0:
        raise SystemExit("V0.17 universal-inference gate failed")
    if report.conditional_reasoning_accuracy < 1.0:
        raise SystemExit("V0.17 conditional-reasoning gate failed")
    if report.causal_retrieval_accuracy < 1.0:
        raise SystemExit("V0.17 causal-retrieval gate failed")
    if report.temporal_order_accuracy < 1.0:
        raise SystemExit("V0.17 temporal-order gate failed")
    if report.negation_accuracy < 1.0:
        raise SystemExit("V0.17 negation gate failed")
    if report.unknown_honesty < 1.0:
        raise SystemExit("V0.17 unknown-honesty gate failed")
    if report.truth_firewall < 1.0:
        raise SystemExit("V0.17 truth-firewall gate failed")
    if report.raw_chat_persisted:
        raise SystemExit("V0.17 persisted raw chat")
    if report.visual_experiences_changed != 0:
        raise SystemExit("V0.17 modified visual training state")
