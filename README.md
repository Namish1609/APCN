# APCN V0.8.2 — Grounded Concept Studio

V0.8.2 fixes the misleading black attention-box rendering bug, restores live APCN concept/feature firing, improves shape candidate calibration, and adds a separate bulk Testing Ground with scoring and confusion matrices.

## Update and run

```bash
cd ~/APCN
git pull
source .venv/bin/activate
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python run_desktop_v0_8.py
```

If dependencies are not installed yet:

```bash
chmod +x install_linux.sh
./install_linux.sh
source .venv/bin/activate
```

## Train tab

The main workflow has three actions:

1. **New / Test Example** — clean prediction-only example; memory does not change.
2. **Teach Visible Example** — learns that exact visible image + sentence once.
3. **Auto Train** — procedural curriculum; every generated episode updates memory.

The Learn/Train preview is intentionally clean. Automatic training can still use background, lighting, position, blur and clutter variation internally.

## Testing Ground

The new dedicated Testing Ground is read-only. It reports:

- color accuracy
- shape accuracy
- joint accuracy
- color confusion matrix
- shape confusion matrix
- per-class recall
- representative failure cases

Choose Clean, Normal, Hard or Stress difficulty and run 100–10,000 balanced samples. The UI shows the episode count before and after the test so you can verify memory was not changed.

## Shape-learning improvement

Previous versions compared words such as `circle`, `square`, `ellipse` and `rectangle` using independently selected feature subsets, which made the resulting scores poorly calibrated. V0.8.2 computes one shared discriminative subspace for all candidates in the comparison and scores every candidate on the same dimensions.

This preserves the existing 23-dimensional memory format, so earlier V0.7/V0.8/V0.8.1 memories can still be loaded.

## Firing display

The Train tab again shows a live APCN firing graph connecting active words/concepts/families to sensory feature nodes. These are concept-graph activations, not neural-network neurons.

## Original dense V0.8 UI

```bash
python run_desktop_v0_8.py --classic
```

## Headless training

```bash
python train_concepts_v0_8.py --episodes 2400 --eval-samples 400
```

## Tests

```bash
python -m unittest discover -s tests -v
```

Full V0.8.2 notes: `README_V0_8_2.md`.
