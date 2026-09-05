# APCN V0.12 — Self-Organizing Perception + Adaptive Correction

V0.12 addresses two persistent V0.11 failure modes by changing the representations that were limiting correction rather than simply increasing training counts:

1. repeated visual confusions such as rectangle↔ellipse and square↔circle;
2. language intent/reference mistakes that could remain sticky after thousands of lifetime cue observations.

There is still **no backpropagation, gradient descent, trainable neural layer or dense learned weight stack**.

## 1. New visual representation

V0.11 still classified vision from a 23-dimensional engineered signal containing channel summaries, scale/position and semantic-style geometric features such as fill/aspect and Hu moments. Error-driven consolidation could improve boundaries but could not recover information discarded by that front end.

V0.12 changes the primary path to:

```text
focused pixels
    ↓
generic translation / scale normalization
+ coarse principal-axis pose normalization
    ↓
generic photometric evidence
  • raw channel distributions
  • continuous channel mean/std
  • brightness-normalized chromatic mean/std
  • joint chromaticity distribution
+
normalized occupancy raster
+
local 3×3 pixel/mask patches
    ↓
unlabeled online competitive patch codebook
    ↓
global + quadrant codeword histograms
    ↓
anonymous V0.12 feature vector
    ↓
compact grounded concept statistics
+ bounded multimodal concept prototypes
    ↓
error-driven consolidation
```

The photometric measurements do not contain named hue bins or rules such as `orange => ...`. They preserve generic continuous information so language can discover whichever dimensions predict a grounded word. Shape classification no longer depends on Hu moments, fill ratio, aspect ratio or an engineered `circle`/`rectangle` detector.

### What the patch learner stores

`SelfOrganizingPatchSensor` maintains at most 24 local patch codewords by default. Each observed patch competes with existing codewords. A sufficiently novel patch can occupy an unused slot; otherwise the winning aggregate prototype is updated with a diminishing local learning rate.

The codebook receives only pixels and the attention mask. It never receives `yellow`, `rectangle`, `circle`, etc. Individual raw patches are discarded after the local update.

## 2. Recent-weighted language correction

V0.11 deliberately retained lifetime aggregate cue/construction counts. This makes knowledge stable, but it also creates a correction problem: after 12,000 observations, a few hundred targeted episodes may be numerically too weak to overturn an old ambiguous construction.

V0.12 therefore keeps two language timescales:

```text
stable lifetime semantic memory
        +
bounded recent construction calibrator
        ↓
intent decision
```

`AdaptiveConstructionCalibrator` stores exponentially decayed intent evidence for abstract constructions such as:

```text
it is the case that <ENTITY> is <REL> the <ENTITY>
can you determine whether <ENTITY> is <REL> the <ENTITY>
```

It stores no raw sentence archive and is bounded by a configurable maximum number of abstract patterns. Recent targeted corrections can therefore matter without deleting the stable long-term language model.

Reference identity continues to use V0.11's bounded discourse entity registry, and V0.11 error signatures for `language_program`, `language_reference` and `language_semantics` are imported into V0.12 so consolidation can immediately target known weaknesses.

## 3. V0.11 migration

The V0.11 visual distributions cannot be mathematically copied into V0.12 because the feature spaces are different.

V0.12 migrates directly:

- language lifetime memory;
- definitions;
- discourse state;
- aggregate error signatures;
- consolidation/language test history.

For vision it retains the previous visual episode count as provenance, but does **not** mix the V0.11 23-D means/variances into the new feature space. A small label-free patch-codebook bootstrap runs automatically, and the first consolidation cycle performs balanced visual grounding if any color/shape concept lacks V0.12 evidence.

This is intentionally explicit in the UI. `5000 legacy visual experiences` and `500 new V0.12 experiences` are not presented as if they were 5500 observations in one compatible coordinate system.

## 4. Desktop workflow

Development branch while V0.12 is being validated:

```bash
cd ~/APCN
git fetch origin
git checkout v0.12-build
git pull
source .venv/bin/activate
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python run_desktop_v0_12.py
```

The UI keeps the existing 1366×768 studio. No additional representation-training tab was added.

The Perception panel adds one compact status block showing:

- patch codewords / maximum;
- local patch updates;
- anonymous V0.12 feature dimension;
- new V0.12 visual evidence;
- legacy V0.11 visual evidence;
- number of ungrounded visual classes.

The existing Consolidation audit additionally shows recent language-correction patterns. The V0.12 consolidation worker writes only to `outputs/v0_12`.

On first launch, if `outputs/v0_12/session_v0_12.json` does not exist but a V0.11 checkpoint does, compatible V0.11 knowledge is imported automatically.

## 5. Headless workflow

Migrate V0.11 knowledge and recalibrate the new visual space:

```bash
python train_v0_12.py \
  --from-v11 \
  --visual 1600 \
  --language 0 \
  --consolidation-cycles 2
```

If a V0.12 checkpoint already exists, omit `--from-v11` to resume it.

## 6. Paired V0.11 vs V0.12 benchmark

Both versions receive exactly the same generated training episodes and identically seeded held-out distributions:

```bash
python benchmark_v0_12.py --train 480 --test 150 --difficulty 0.68
```

An early V0.12 patch-only descriptor fixed the shape errors but regressed color, which is why it was not released. After restoring generic continuous photometric information, the matched smoke benchmark produced:

```text
                     V0.11       V0.12
color accuracy       99.3%       100.0%
shape accuracy       84.7%       100.0%
joint accuracy       84.0%       100.0%

V0.11 shape errors:
  square -> circle        9
  ellipse -> rectangle    9
  rectangle -> ellipse    5

V0.12 shape errors:       0
V0.12 color errors:       0
```

This is evidence on one deterministic paired benchmark, not proof of general visual intelligence. V0.12 also includes a harder three-seed validation at difficulty 0.82:

```bash
python validate_v0_12.py
```

The CI release gate requires strong absolute V0.12 color/shape/joint accuracy as well as no material regression relative to V0.11.

## 7. Memory scaling

V0.12 still does not archive all images, raw patches or sentences.

Long-term memory is approximately:

```text
small patch codebook
+ token sufficient statistics
+ bounded concept prototype banks
+ lifetime language cue/construction aggregates
+ bounded recent language calibration patterns
+ aggregate error signatures
+ sparse unified concept graph
```

Experience count and retained knowledge size therefore remain deliberately separate.

## 8. Scientific boundary

V0.12 has substantially less semantic handcrafting in shape perception, but it still has generic inductive biases:

- attention/focus masks;
- translation and scale normalization;
- principal-axis pose normalization for sufficiently anisotropic objects;
- fixed local patch size;
- generic photometric statistics;
- spatial pooling.

It is not raw-pixel intelligence with zero prior structure, and a synthetic shape benchmark does not establish scalability to unrestricted real-world vision.

The release question is narrower and falsifiable: **does replacing the lossy handcrafted geometry front end plus adding bounded recent correction reduce the exact persistent errors while retaining compact, non-gradient learning?**
