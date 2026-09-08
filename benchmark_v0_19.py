from __future__ import annotations

import json

from apcn_v19.benchmark import run_language_math_benchmark


def main() -> int:
    report = run_language_math_benchmark()
    print(json.dumps(report.to_dict(), indent=2))
    metrics = [
        report.basic_addition_accuracy,
        report.basic_subtraction_accuracy,
        report.english_recombination_accuracy,
        report.reverse_subtraction_accuracy,
        report.word_number_accuracy,
        report.invented_operator_one_shot_accuracy,
        report.invented_subtraction_one_shot_accuracy,
        report.unknown_operator_honesty,
        report.persistence_accuracy,
        report.v018_fallback_accuracy,
        report.no_backprop_contract,
    ]
    return 0 if all(abs(x - 1.0) < 1e-12 for x in metrics) else 1


if __name__ == "__main__":
    raise SystemExit(main())
