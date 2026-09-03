# APCN V0.10.2 — Visible Training Activation Layout Hotfix

V0.10.2 fixes the V0.10.1 training-page activation display. The earlier patch appended `SpikingGraph` widgets by searching for the last `QFrame` in each page, which could place them in a squeezed or effectively invisible area at 1366×768.

## What changed

The firing/activation graph is now a first-class, resizable region in every training tab:

- **Perception**: scene on top, `PERCEPTION FIRING / SPIKING — CURRENT SAMPLE` directly below it.
- **Language**: semantic program comparison on top, `LANGUAGE FIRING / SPIKING — CURRENT SENTENCE` directly below it.
- **Definitions**: concept status on top, `DEFINITION FIRING / SPIKING — ACTIVE DEPENDENCY PATH` directly below it.
- **Ask APCN** retains its query concept activation graph from V0.10.1.

These graphs show sparse APCN feature/cue/concept/operator activation. They are diagnostic visualizations and are **not** biological neurons or a spiking neural network.

The UI remains designed for 1366×768 and uses Qt splitters, so the activation area can be resized manually.

## Run

```bash
cd ~/APCN
git pull
source .venv/bin/activate
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python run_desktop_v0_10.py
```

The normal V0.10 launcher now opens V0.10.2. You can also run:

```bash
python run_desktop_v0_10_2.py
```

## Architecture status

APCN remains non-neural in these prototypes: no trainable neural layers, gradient descent, backpropagation, optimizer or dense learned weight matrices are used by the V0.7–V0.10 concept/language/definition learners. The perception front end still contains engineered generic measurements, which remains an important research limitation.
