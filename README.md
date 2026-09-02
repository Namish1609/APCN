# APCN V0.7 — Procedural Teacher + Grounded Concept Discovery

V0.7 implements the training mechanism discussed after V0.6. It is designed for a **headless Linux server** and does not require PyQt or a display server.

## What V0.7 tests

The procedural teacher automatically creates objects from independently varied factors:

- colors: yellow, red, green, blue, purple, orange
- shapes: circle, square, ellipse, triangle, rectangle
- randomized size, position, rotation
- randomized brightness/background/noise/blur
- optional distractor objects
- varied English templates such as `this is a yellow circle`, `the circle is yellow`, `look at the yellow circle`

APCN receives only:

1. pixels,
2. a **joint-attention mask** indicating what the teacher is pointing at,
3. the English utterance.

`teacher_metadata` exists for evaluation but `GroundedConceptLearner.train_episode()` never reads it.

The learner is **not told** that `yellow` is a color word or `circle` is a shape word. It accumulates compact sufficient statistics for every token and asks which anonymous sensory dimensions are stable/discriminative when that token occurs.

Because all color × shape combinations occur, shape changes under a fixed color and color changes under a fixed shape. This breaks the shortcut correlation and lets different words select different sensory subspaces.

## Important scientific boundary

V0.7 does **not** yet discover vision directly from raw pixels with zero engineered processing. `AnonymousVisualSensor` computes a small generic vector of channel statistics, location/scale measurements and image moments. The learner sees only anonymous dimensions `f000...f022`; it does not receive the human diagnostic groups.

So the experiment is:

> Can controlled language experience discover which *existing anonymous sensory dimensions* matter for a word?

It is not yet:

> Can APCN invent all primitive visual features from raw photons?

That should be a later front-end experiment.

## Memory scaling

No individual training images are retained by the concept learner. Each token stores only:

- count
- feature sum
- squared-feature sum

For feature dimension `D` and vocabulary size `V`, semantic grounding memory is approximately `O(V × D)`, not `O(number_of_training_episodes)`.

## Linux installation

```bash
git clone https://github.com/Namish1609/APCN.git
cd APCN
chmod +x install_linux.sh
./install_linux.sh
source .venv/bin/activate
```

Manual alternative:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For Ubuntu/Debian, if `python3 -m venv` is missing:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
```

## Train

```bash
python train_concepts_v0_7.py --episodes 2400 --eval-samples 400
```

Outputs:

```text
outputs/v0_7/concept_memory_v0_7.json
outputs/v0_7/training_report_v0_7.json
```

## Generalization test

```bash
python test_generalization_v0_7.py outputs/v0_7/concept_memory_v0_7.json --samples 500
```

This creates new held-out objects with stronger nuisance variation and asks the learned word models to identify color and shape.

## Inspect what a word learned

```bash
python inspect_concepts_v0_7.py outputs/v0_7/concept_memory_v0_7.json yellow circle this is
```

`diagnostic_signal_mass` is **human-only instrumentation**. The learner never receives labels such as `channel_signal` or `geometry_signal` during training.

Expected qualitative result:

- `yellow` places most relevance on channel-signal dimensions;
- `circle` places most relevance on geometry dimensions;
- `this` / `is` have much weaker visual concept quality.

## Generate a preview image on a headless server

```bash
python generate_training_preview_v0_7.py
```

It writes:

```text
outputs/v0_7/training_preview.jpg
```

You can copy that image from the server with `scp` or open it through whatever file browser your cloud provider offers.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Architecture

```text
PROCEDURAL TEACHER
    │
    ├── rendered pixels
    ├── joint-attention mask
    └── English utterance
           │
           ▼
ANONYMOUS VISUAL SENSOR
           │
           └── f000 ... f022
                    │
                    ▼
GROUNDED CONCEPT LEARNER
    ├── global running statistics
    ├── token running statistics
    ├── discriminative/invariant feature weighting
    └── no episode archive
                    │
                    ▼
CURRICULUM ENGINE
    ├── factorial bootstrap
    ├── minimal contrasts
    ├── nuisance randomization
    └── active learning of weakest content word
```

## Why the teacher can know the answer

The synthetic teacher is the environment, not the AI. It is allowed to know that it rendered a yellow circle so it can provide a learning signal. The learner is deliberately prevented from reading that metadata. This is analogous to a human teacher pointing at an object and saying a word.

The next step after V0.7 is to replace synthetic-only teaching with mixed sources:

1. procedural scenes,
2. 3D rendered scenes,
3. real camera frames + human click/box for joint attention,
4. corrections and active questions from APCN itself.

## Teach from a real camera/image

V0.7 can update the same memory from a real image. The user/teacher supplies joint attention either as a mask or a bounding box.

Bounding-box example:

```bash
python teach_from_image_v0_7.py \
  --image camera_frame.jpg \
  --bbox 120,80,180,180 \
  --utterance "this is a yellow ball"
```

Mask example:

```bash
python teach_from_image_v0_7.py \
  --image camera_frame.jpg \
  --mask yellow_ball_mask.png \
  --utterance "this is a yellow ball"
```

This does **not** mean a single example is sufficient. The value is that real examples enter exactly the same cross-situational learner as synthetic experiences, so synthetic bootstrap and real-world correction can be mixed.
