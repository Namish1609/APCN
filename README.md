# APCN V0.8 — Interactive Grounded Concept Laboratory

V0.8 keeps the V0.7 procedural grounded-language trainer and restores a large PyQt6 desktop interface similar to the earlier APCN Windows laboratory. It is designed to run on Linux desktops/cloud servers with GUI forwarding or remote desktop, while retaining headless CLI training.

## Quick start — Linux UI

```bash
git pull
chmod +x install_linux.sh
./install_linux.sh
source .venv/bin/activate
python run_desktop_v0_8.py
```

## What is new

- large resizable PyQt6 desktop UI
- live procedural teacher/camera scene and joint-attention box
- live sparse concept/feature firing graph
- automatic run/pause training with progress
- manual teaching on generated examples
- real-image teaching by dragging a focus rectangle
- held-out evaluation, concept inspector and activity log
- automatic visual-vs-structural word role inference
- automatic unlabeled concept-family discovery
- resumable V0.8 checkpoints
- headless training remains supported

The full V0.8 notes are in `README_V0_8.md`.

## Desktop command box

```text
train 100
show yellow circle
inspect yellow
test 200
save
load
```

## Headless training

```bash
chmod +x install_headless_linux.sh
./install_headless_linux.sh
source .venv/bin/activate
python train_concepts_v0_8.py --episodes 2400 --eval-samples 400
```

## Generalization

```bash
python test_generalization_v0_8.py outputs/v0_8/concept_memory_v0_8.json --samples 500 --difficulty 0.92
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Scientific boundary

V0.8 still uses V0.7's 23-dimensional generic visual front-end. The English learner is not told which dimensions mean color or geometry, but the low-level feature extractor itself is still engineered. The next research target is self-organizing raw-sensory feature creation.
