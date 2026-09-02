# APCN V0.8.2 — Grounded Concept Studio

V0.8.2 focuses on making the training process visually trustworthy and measurable.

## Main fixes

### 1. Black square over every focused object

The V0.8.1 canvas drew the attention bounding rectangle while the QPainter still had the black canvas brush active. `drawRect()` therefore filled the whole attention box black and covered the real object. The learner still received the correct image/mask; the display was wrong.

V0.8.2 explicitly sets `Qt.NoBrush` before drawing the dashed attention outline.

### 2. Clean visual examples vs difficult training scenes

The Train tab now generates clean, solid-background prediction examples by default so you can visually confirm that `blue circle` really is a blue circle. Automatic training still uses background/lighting/noise/clutter variation internally because those nuisance factors help prevent shortcuts.

Hard visual conditions are tested explicitly in the Testing Ground.

### 3. Shape score calibration

The old classifier let each candidate word choose a different sensory subspace. Scores from `circle`, `square`, `ellipse`, etc. were therefore not always directly comparable.

V0.8.2 computes one shared discriminative subspace for the candidate family and scores every candidate on the same dimensions. It is an online classical-statistics/LDA-like calculation; no gradients or neural layer were added.

The 23-dimensional memory format is retained, so V0.7/V0.8/V0.8.1 checkpoints can still be loaded.

### 4. Firing visualization restored

The Train tab again shows a live neuron-like firing graph. These nodes are APCN concept/feature activation nodes, not biological or neural-network neurons.

### 5. Dedicated Testing Ground

Bulk tests are prediction-only and guarantee that episode count is unchanged. The tab reports:

- color accuracy
- shape accuracy
- joint accuracy
- color confusion matrix
- shape confusion matrix
- per-class recall
- representative failures

The benchmark cycles through color × shape combinations to avoid random class imbalance.

## Update and run

```bash
cd ~/APCN
git pull
source .venv/bin/activate
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python run_desktop_v0_8.py
```

## Recommended training workflow

1. On **Train**, choose a specific `Color` and `Shape` once and click **New / Test Example**. Verify the rendered object matches the teacher truth.
2. Use **Teach Visible Example** only to understand one-step learning.
3. Set Auto Train to 500–3000 episodes and train automatically.
4. Go to **Testing Ground**. Run 500 samples at `Clean`, `Normal`, `Hard`, then `Stress` difficulty.
5. Inspect the shape confusion matrix. If `circle → square` or `ellipse → rectangle` dominates, continue targeted training and compare the matrix again.
6. Save memory from the **Concepts** tab.

## What each mode means

- **New / Test Example**: predicts only; memory frozen.
- **Teach Visible Example**: adds one displayed image + sentence to memory.
- **Auto Train**: generated curriculum; every episode updates memory.
- **Testing Ground**: read-only benchmark; memory must not change.

## Regression tests

```bash
python -m unittest discover -s tests -v
```

V0.8.2 includes tests ensuring clean preview and bulk evaluation are read-only, shared candidate relevance is learned, and V0.8.2 checkpoints reload correctly.
