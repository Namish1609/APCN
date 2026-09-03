# APCN V0.12 — Self-Organizing Perception + Adaptive Correction

Current release: **0.12.0**.

V0.12 addresses the persistent V0.11 error loop by changing two limiting representations rather than simply increasing training volume:

- visual shape concepts now use a less-handcrafted pixel/patch representation instead of the old 23-D Hu/fill/aspect geometry summary;
- language adds a bounded recent-evidence construction calibrator so targeted corrections can matter after thousands of lifetime observations.

It remains non-neural in the current implementation: **no backpropagation, gradient descent, trainable neural layers or dense learned weight stack**.

## What changed

The V0.12 perception path is:

```text
focused pixels
    ↓
generic normalization
    ↓
generic photometric evidence
+ normalized occupancy raster
+ local pixel/mask patches
    ↓
unlabeled online competitive patch codebook
    ↓
spatial codeword representation
    ↓
compact concept statistics + bounded prototypes
```

The patch codebook is bounded and label-free. It never receives words such as `circle`, `rectangle`, `yellow` or `orange`. Shape classification no longer relies on Hu moments, fill ratio or aspect ratio.

Language now combines:

```text
stable lifetime cue/construction memory
        +
bounded recent construction correction
```

so known errors such as `ASSERT→GOAL`, `QUERY→ASSERT` and nested `NEGATE>ASSERT→NEGATE>GOAL` can receive meaningful targeted corrective evidence without deleting long-term knowledge.

V0.11's error memory, discourse entity registry, unified concept graph and automatic consolidation loop remain in place.

## Update and run

```bash
cd ~/APCN
git checkout main
git pull
source .venv/bin/activate
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python run_desktop_v0_12.py
```

The UI remains based on the existing **1366×768** studio. V0.12 does not add another cluttered representation-training tab. The Perception panel shows a compact representation status and the existing Consolidation page shows learning priorities, memory audit, concept graph, recent language correction state and before→after curves.

## Migration from V0.11

On first V0.12 launch, if `outputs/v0_12/session_v0_12.json` does not exist but `outputs/v0_11/session_v0_11.json` does, compatible V0.11 knowledge is imported automatically.

Directly migrated:

- language lifetime memory;
- definitions;
- discourse state;
- aggregate error signatures;
- consolidation/test history.

The V0.11 visual means/variances are **not** copied into V0.12 because the feature spaces are incompatible. Their episode count is retained as provenance, a small label-free patch-codebook bootstrap runs, and the first consolidation cycle automatically builds balanced V0.12 visual evidence before using known confusion signatures for targeted contrasts.

After launch, the recommended first action is:

```text
Consolidation → Run 1 Automatic Consolidation Cycle
```

## Tested perception result

Paired V0.11/V0.12 tests use exactly the same training scenes and identically seeded held-out distributions.

Smoke benchmark, 480 training experiences / 150 tests / difficulty 0.68:

```text
                     V0.11       V0.12
color                99.3%       100.0%
shape                84.7%       100.0%
joint                84.0%       100.0%
```

V0.11 produced repeated `square→circle`, `ellipse→rectangle` and `rectangle→ellipse` errors. V0.12 produced no visual errors in that matched run.

Hard three-seed gate, 720 training experiences / 180 tests per seed / difficulty 0.82:

```text
mean                 V0.11       V0.12
color                99.44%      98.15%
shape                90.93%      99.81%
joint                90.37%      97.96%
```

Across 540 V0.12 hard held-out scenes there was one shape error (`triangle→ellipse`); the repeated rectangle↔ellipse and square↔circle failures were absent. V0.12 therefore trades a small amount of hard-regime color accuracy for a much larger shape/joint improvement in this synthetic benchmark.

These results are **not** proof of general real-world vision or general intelligence; they are evidence that the new representation addresses the specific synthetic perception bottleneck that persisted in V0.11.

## Headless experiments

Migrate V0.11 and continue:

```bash
python train_v0_12.py \
  --from-v11 \
  --visual 1600 \
  --language 0 \
  --consolidation-cycles 2
```

Run the matched benchmark:

```bash
python benchmark_v0_12.py --train 480 --test 150 --difficulty 0.68
```

Run the three-seed hard gate:

```bash
python validate_v0_12.py
```

Run all regressions:

```bash
python -m unittest discover -s tests -v
```

## Memory model

V0.12 still does not retain every image, local patch or sentence. Long-term state consists primarily of:

```text
bounded patch codebook
+ visual sufficient statistics
+ bounded visual concept prototypes
+ lifetime language cue/construction aggregates
+ bounded recent language corrections
+ aggregate error signatures
+ sparse concept graph
+ bounded discourse working memory
```

See `README_V0_12.md` for the architecture, migration boundary, benchmarks and scientific limitations.
