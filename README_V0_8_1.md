# APCN V0.8.1 — Guided Grounded Concept Training

The default desktop UI is now a simplified **1366×768-friendly guided trainer**. The original advanced V0.8 UI is still available with `--classic`.

## Update and run

```bash
cd ~/APCN
git pull
source .venv/bin/activate
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python run_desktop_v0_8.py
```

## Training workflow

The default **Learn** tab deliberately has only three primary actions:

1. **New / Test Example** — generate a visible example and let APCN predict it. This does **not** change memory.
2. **Teach This Visible Example** — learn that exact visible image + teacher sentence once. The episode counter increases by exactly one.
3. **Auto Train** — generate and learn the selected number of procedural lessons. The screen only samples occasional lessons so the UI stays readable; every generated lesson is still learned.

After auto-training, click **New / Test Example** again. That is the cleanest way to see whether APCN generalizes without accidentally training on the test.

The mode banner always says either:

- `PREVIEW / TEST — NOT LEARNING`
- `LEARNING — MEMORY IS CHANGING`
- `TRAINING COMPLETE`

so training and testing are visually distinct.

## Purple object appearing black on Xvfb/noVNC

V0.8.1 changes the Qt rendering path from direct `Format_BGR888` display to an explicit:

```text
OpenCV BGR → RGB conversion → QImage Format_RGB888
```

This is more portable across Xvfb/x11vnc/noVNC stacks and fixes the wrong/black color rendering seen on some virtual Linux desktops.

Visible user-generated examples also default to low difficulty with no distractors so that the displayed color and shape are easy to verify. Automatic curriculum training still uses varied difficulty internally.

## Inspect tab

Use **Inspect** only when you want details. It contains:

- learned words and concept quality
- inferred visual/structural role
- discovered unlabeled word families
- save/load memory
- held-out evaluation

These controls were intentionally removed from the main Learn screen.

## Real image / manual tab

1. Load an image.
2. Drag a box around one object.
3. Type a sentence such as `this is a yellow ball`.
4. Click **Teach focused object**.

This updates the same persistent concept memory as synthetic training.

## Original V0.8 UI

If you need the old dense laboratory:

```bash
python run_desktop_v0_8.py --classic
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The cognitive backend is unchanged by this UI cleanup. V0.8.1 is primarily a usability/rendering fix on top of the V0.8 learner and session architecture.
