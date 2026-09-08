# APCN V0.19 — Online English + Arithmetic

V0.19 deliberately pauses the world-model expansion. The immediate question is narrower and falsifiable:

> Can APCN acquire useful English arithmetic constructions online and generalize them to unseen operands and newly taught words without backpropagation, gradient descent, an external LLM, or a pretrained language model?

## Why this milestone exists

V0.18 improved explicit semantic memory, proof, causal retrieval, concept identity, and natural realization, but real desktop testing showed that this can become an endless architecture project while the language core remains too brittle for ordinary use.

V0.19 therefore treats language competence itself as the product under test. The world model remains available only through the V0.18 fallback path; it is not the focus of this release.

## Learning mechanism

`OnlineConstructionArithmeticV19` uses a small semantic hypothesis set:

- `ADD(a,b)`
- `SUB(a,b)`
- `SUB(b,a)` for constructions such as `subtract 3 from 10`

A teaching example such as:

```text
remember that 2 dax 3 equals 5
```

is not stored as an answer lookup table. APCN evaluates which candidate semantic operations are consistent with the example. Here only addition is consistent. Surface unigram/bigram/trigram evidence from `dax` and its construction is then associated with the selected semantic operation.

A later query:

```text
what is 11 dax 8?
```

contains new operands. Learned construction evidence identifies `ADD`; the arithmetic executor computes `11 + 8` at runtime.

No gradient is computed and no model weights are trained.

## Bootstrap English curriculum

V0.19 starts with a compact set of English arithmetic demonstrations applied through the same online learning API. They include examples of:

- `plus`
- `add ... and ...`
- `sum of ... and ...`
- `minus`
- `subtract ... from ...`
- `take ... away from ...`
- `+` and `-`
- wrappers such as `what is`, `how much is`, and `calculate`

The parser does not contain branches such as `if word == "plus": add`. The words gain operation evidence from demonstrations.

The benchmark then uses held-out numbers, recombined wrappers, and invented operator words learned only during the benchmark.

## Finite V0.19 gate

V0.19 must score 100% on:

1. unseen-number addition;
2. unseen-number subtraction;
3. English wrapper/operator recombination;
4. reverse-order subtraction constructions;
5. basic written number words;
6. one-shot invented addition word learning;
7. one-shot invented subtraction word learning;
8. honesty on an unknown arithmetic word;
9. persistence of newly learned arithmetic language;
10. preservation of the V0.18 semantic fallback;
11. the no-backprop/no-external-model architecture contract.

Run:

```bash
python -m unittest tests.test_v0_19 -v
python benchmark_v0_19.py
```

## Desktop

```bash
python run_desktop_v0_19.py
```

Suggested clean-session sequence:

```text
what is 37 plus 58?
what is 91 minus 47?
subtract 7 from 30
what is seven plus five
what is 11 dax 8?
remember that 2 dax 3 equals 5
what is 11 dax 8?
remember that 9 nerk 4 equals 5
please calculate 20 nerk 7
what is 9 florp 2?
```

Expected behavior:

- known constructions calculate correctly with unseen operands;
- `dax` becomes addition from the one example;
- `nerk` becomes subtraction from the one example;
- `florp` remains unknown rather than being guessed.

## Scientific boundary

Passing V0.19 is **not** a claim that APCN is an LLM or has broad English competence. It proves a smaller but more useful capability than previous transcript patches: online induction and compositional reuse of arithmetic language without backpropagation.

The next language milestones should expand the semantic hypothesis space and construction learner only after this finite gate is stable: multiplication/division, comparison, variables, multi-step arithmetic, then broader sentence semantics and paraphrase learning. World-model integration should resume only after the language core reaches a useful benchmark level.
