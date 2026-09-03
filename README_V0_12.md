# APCN V0.12 — Self-Organizing Perception

V0.12 addresses the persistent V0.11 color/shape failure loop by changing the sensory representation rather than only adding more examples or changing the classifier.

## Why V0.12 exists

V0.11 still classified vision from a 23-dimensional engineered signal containing channel summaries, position/scale, fill/aspect and Hu-moment geometry. Error-driven consolidation could improve decision boundaries, but it could not recover visual information discarded by that front end.

V0.12 changes the primary perception path to:

```text
focused pixels
    ↓
generic pose/scale normalization
    ↓
raw per-channel pixel distributions
+ normalized occupancy raster
+ local 3x3 pixel/mask patches
    ↓
unlabeled online competitive patch codebook
    ↓
spatial codeword histogram
    ↓
anonymous V0.12 feature vector
    ↓
compact grounded concept statistics
+ bounded multimodal concept prototypes
    ↓
error-driven consolidation
```

There is still **no backpropagation, gradient descent, neural layer or dense trainable weight stack**.

## What is actually learned

`SelfOrganizingPatchSensor` maintains at most 24 local patch codewords. A patch is assigned to the closest codeword; sufficiently novel patches occupy unused slots, otherwise the winning prototype is updated locally with a diminishing learning rate.

The codebook receives only image pixels and the attention mask. It does not receive `yellow`, `rectangle`, `circle`, etc.

The object descriptor then contains:

- per-channel pixel histograms;
- a low-resolution normalized occupancy raster;
- global + quadrant histograms over learned patch-codeword IDs.

The learner still sees anonymous dimensions (`f000`, `f001`, ...).

## Removed as primary classifier features

V0.12 does not use the V0.7-V0.11 Hu-moment/fill/aspect 23-D descriptor as the primary classifier input.

That does **not** mean V0.12 has zero inductive bias. It still uses:

- an attention/focus mask;
- translation and scale normalization;
- coarse principal-axis pose normalization for anisotropic objects;
- local fixed-size patches;
- spatial pooling.

These are generic vision biases, not learned semantic labels. A later version can reduce these assumptions further.

## V0.11 migration

The V0.11 visual distributions cannot be copied into V0.12 because the feature spaces are mathematically different.

V0.12 therefore migrates:

- language memory;
- definitions;
- discourse state;
- error signatures;
- consolidation history;
- the old visual episode count as provenance.

It does **not** mix V0.11 23-D means/variances into V0.12 statistics.

On first desktop launch, if `outputs/v0_12/session_v0_12.json` does not exist but `outputs/v0_11/session_v0_11.json` does, compatible V0.11 knowledge is imported automatically and a small label-free patch-codebook bootstrap is run.

The first V0.12 consolidation cycle also performs balanced visual grounding automatically if the new feature space has not yet covered every color/shape concept.

## Desktop

```bash
cd ~/APCN
git checkout v0.12-build
git pull
source .venv/bin/activate
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python run_desktop_v0_12.py
```

The UI remains based on the 1366×768 V0.11 studio. There is no additional cluttered representation-training tab. The Perception page adds a compact representation status line showing:

- learned patch codewords;
- patch updates;
- V0.12 feature dimension;
- new V0.12 visual evidence;
- legacy V0.11 visual evidence;
- missing grounded visual classes.

## Headless migration + training

```bash
python train_v0_12.py \
  --from-v11 \
  --visual 1600 \
  --language 0 \
  --consolidation-cycles 2
```

If you already have a V0.12 checkpoint, omit `--from-v11`; the command resumes from `outputs/v0_12`.

## Paired benchmark

V0.12 includes a paired benchmark where V0.11 and V0.12 receive exactly the same training scenes and are evaluated on identically seeded held-out scenes:

```bash
python benchmark_v0_12.py --train 1800 --test 600 --difficulty 0.86
```

Do not judge V0.12 only by one aggregate score. Inspect the `shape_confusions` and `color_confusions` maps, especially:

```text
rectangle -> ellipse
ellipse -> rectangle
square -> circle
circle -> square
yellow -> orange
orange -> yellow
```

## Memory scaling

V0.12 still does not archive training images or raw local patches.

Long-term visual memory is bounded by:

```text
small patch codebook
+ token sufficient statistics
+ bounded concept prototype banks
+ aggregate error signatures
```

The codebook stores at most the configured number of aggregate patch prototypes; individual patches are discarded immediately after their local update.

## Scientific release gate

V0.12 should only replace V0.11 on `main` if:

1. the full regression suite passes;
2. read-only testing provably does not update the patch codebook;
3. visual memory remains bounded;
4. moderate-difficulty color/shape learning stays above chance;
5. paired benchmarking shows whether the persistent confusion families improve or regress.

The benchmark result is evidence, not a promise. If V0.12 does not improve the hard shape confusions, the correct response is to revise the representation—not hide the result by increasing training counts.
