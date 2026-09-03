from __future__ import annotations

import json

from apcn_v12.benchmark import run_paired_benchmark


def main() -> int:
    seeds = (12012, 12029, 12047)
    rows = []
    for seed in seeds:
        rep = run_paired_benchmark(
            train_experiences=720,
            test_samples=180,
            difficulty=.82,
            seed=seed,
        )
        rows.append(rep.to_dict())

    def mean(path):
        vals = []
        for row in rows:
            x = row
            for key in path:
                x = x[key]
            vals.append(float(x))
        return sum(vals) / len(vals)

    summary = {
        "seeds": list(seeds),
        "train_per_seed": 720,
        "test_per_seed": 180,
        "difficulty": .82,
        "v011": {
            "color": mean(("v011", "color_accuracy")),
            "shape": mean(("v011", "shape_accuracy")),
            "joint": mean(("v011", "joint_accuracy")),
        },
        "v012": {
            "color": mean(("v012", "color_accuracy")),
            "shape": mean(("v012", "shape_accuracy")),
            "joint": mean(("v012", "joint_accuracy")),
        },
        "runs": rows,
    }
    print(json.dumps(summary, indent=2))

    # Release-quality guardrails. These are deliberately absolute as well as
    # comparative so a weak V0.11 run cannot make a weak V0.12 run look good.
    v11 = summary["v011"]
    v12 = summary["v012"]
    failures = []
    if v12["color"] < .94:
        failures.append(f"mean V0.12 color accuracy too low: {v12['color']:.3f}")
    if v12["shape"] < .92:
        failures.append(f"mean V0.12 shape accuracy too low: {v12['shape']:.3f}")
    if v12["joint"] < .90:
        failures.append(f"mean V0.12 joint accuracy too low: {v12['joint']:.3f}")
    if v12["shape"] + .01 < v11["shape"]:
        failures.append("V0.12 shape representation regressed relative to V0.11")
    if v12["joint"] + .02 < v11["joint"]:
        failures.append("V0.12 joint performance regressed materially relative to V0.11")

    if failures:
        print("\nV0.12 HARD GATE FAILED")
        for item in failures:
            print(f"- {item}")
        return 1
    print("\nV0.12 HARD GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
