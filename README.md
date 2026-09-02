# APCN V0.8.1 — Guided Grounded Concept Training

The default desktop UI is now a simplified **1366×768-friendly** trainer for Linux/Xvfb/noVNC. The cognitive backend remains the V0.8 learner/session architecture.

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

## How to train

The Learn tab has three primary actions:

1. **New / Test Example** — APCN predicts the visible example. Memory does **not** change.
2. **Teach This Visible Example** — the exact visible pixels + teacher sentence are learned once.
3. **Auto Train** — the procedural teacher generates and learns a batch of grounded lessons automatically.

After auto-training, press **New / Test Example** to check generalization without teaching on the test.

The large mode banner explicitly shows whether memory is changing.

## V0.8.1 UI changes

- default window sized for 1366×768
- much less cluttered Learn screen
- tutorial always visible
- training/test state clearly separated
- advanced inspection moved to the Inspect tab
- real-image teaching moved to its own tab
- visible examples use low difficulty/no distractors by default
- fixed virtual-desktop color rendering by using explicit OpenCV BGR → RGB → `QImage.Format_RGB888`

Full notes: `README_V0_8_1.md`.

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

## Scientific boundary

V0.8.1 still uses V0.7/V0.8's 23-dimensional generic visual front-end. The language learner is not told which dimensions mean color or geometry, but the primitive visual feature extractor is still engineered. The next research target remains online self-organizing raw-sensory feature creation.
