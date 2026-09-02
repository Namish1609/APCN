# APCN V0.9 — Grounded Semantic Language Acquisition

V0.9 moves APCN beyond isolated color/shape words into **utterance → contextual semantic program** learning.

## New research capability

V0.9 learns cue mappings from procedural grounded language/world demonstrations for:

- entity composition: color + shape → entity reference
- binary spatial relations (`R0`, `R1`, `R2` internally)
- sentence intent: `ASSERT`, `QUERY`, `GOAL`
- conjunction / grouping
- temporal sequence
- negation
- contextual reference to the currently salient entity
- arbitrary vocabulary tests (`zorp`, `ka`, `vo`, `blicket`, etc.) proving literal English strings are not required
- persistent compact cue/feature statistics
- read-only semantic benchmarks with failures, confusion matrices and test-history graphs

Example learned interpretation:

```text
put the yellow circle and the green triangle inside the red square

GROUP
  GOAL
    R0(C0:S0, C1:S1)
  GOAL
    R0(C2:S2, C1:S1)
```

The surface word `inside` is not hardcoded to `R0` in the learner. The procedural teacher supplies language-independent semantic demonstrations, and cross-situational contrast learns which cue predicts which semantic feature.

## Scientific boundary

This is **not full English understanding**. V0.9 still has engineered internal semantic primitives (`ASSERT`, `QUERY`, `GOAL`, `GROUP`, `SEQUENCE`, `NEGATE`, `RELATION`). The learned part is how surface cues, order and context map onto those primitives.

The language-world demonstrations are also synthetic. The next step is to connect these semantic episodes to APCN's observed world/belief model and later to definitions and textbook learning.

## Linux UI

```bash
cd ~/APCN
git pull
source .venv/bin/activate
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python run_desktop_v0_9.py
```

The window remains designed for **1366×768**.

Tabs:

1. **Perception** — V0.8.2 visual grounding retained.
2. **Language + Context** — generate, test and teach semantic sentences and run the staged language curriculum.
3. **Testing Ground** — perception or language tests. The upper area contains confusion matrices. The lower area is split exactly into **Failures** and **Graphs**. There are no redundant score cards.
4. **Concept Memory** — learned perceptual concepts and semantic cue associations.

## Recommended training

Perception, if needed:

```text
Perception → Auto Train Batch → 2000–3000
```

Language:

```text
Language + Context → Auto Train Language Batch → 2200–3000
```

The language curriculum grows in stages:

```text
0–399       lexical relations
400–799     statement / question / command intent
800–1199    entity composition + conjunction
1200–1599   sequence + negation
1600+       mixed language + contextual reference
```

After each stage, run **Testing Ground → Language semantics**. Tests never call the learner's `observe()` method.

## CLI

```bash
python train_language_v0_9.py --episodes 2600 --test-samples 500
```

Outputs:

```text
outputs/v0_9/semantic_memory_v0_9.json
outputs/v0_9/session_v0_9.json
outputs/v0_9/language_report_v0_9.json
```

Re-test a saved memory:

```bash
python test_language_v0_9.py outputs/v0_9/semantic_memory_v0_9.json --samples 500
```

## Tests

```bash
python -m unittest discover -s tests -v
```

V0.9 unit tests verify relation grounding, intent contrast, group/sequence/negation parsing, arbitrary vocabulary learning, read-only evaluation, benchmark thresholds and persistence.

## Reference controlled benchmark

A deterministic staged run produced:

| curriculum steps | exact program | intent | relation | operator |
| ---: | ---: | ---: | ---: | ---: |
| 200 | 23.3% | 33.3% | 99.3% | 70.0% |
| 400 | 23.3% | 33.3% | 99.7% | 70.0% |
| 800 | 70.0% | 100% | 99.7% | 70.0% |
| 1200 | 80.0% | 100% | 98.7% | 80.0% |
| 1600 | 98.7% | 100% | 100% | 98.7% |
| 2200 | 100% | 100% | 100% | 100% |
| 3000 | 100% | 100% | 100% | 100% |

These are **synthetic controlled-language benchmark results**, not open-domain English accuracy.

The separate held-out-template benchmark is intentionally harder because it introduces constructions that were not in the training templates. That benchmark remains materially weaker and is a useful target for V0.10 construction induction.
