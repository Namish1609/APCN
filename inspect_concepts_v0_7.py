#!/usr/bin/env python3
from __future__ import annotations

import argparse
from apcn_v07.learner import GroundedConceptLearner


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("memory", nargs="?", default="outputs/v0_7/concept_memory_v0_7.json")
    p.add_argument("words", nargs="*")
    args = p.parse_args()

    learner = GroundedConceptLearner.load(args.memory)
    words = args.words or sorted(learner.token_stats)
    for word in words:
        profile = learner.token_profile(word)
        profile["diagnostic_signal_mass"] = learner.diagnostic_group_mass(word)
        print(profile)


if __name__ == "__main__":
    main()
