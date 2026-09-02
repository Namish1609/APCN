# APCN V0.8 — Interactive Grounded Concept Laboratory

V0.8 keeps the V0.7 procedural grounded-language trainer and restores a large PyQt6 desktop interface similar to the earlier APCN Windows laboratory. It is designed to run on Linux desktops/cloud servers with GUI forwarding or remote desktop, while retaining headless CLI training.

## What is new

- large resizable PyQt6 desktop UI
- live procedural teacher/camera scene
- visible joint-attention focus box
- live sparse concept/feature firing graph
- automatic training with run/pause and progress
- manual teaching utterances on the current focused object
- real-image teaching: load a photo and drag a focus rectangle
- held-out evaluation from the UI
- concept inspector and persistent activity log
- automatic visual-vs-structural word role inference
- automatic discovery of token families from relevance-profile similarity
- resumable training and V0.8 memory checkpoints
- headless install/training remains supported

## Scientific boundary

V0.8 still uses the V0.7 `AnonymousVisualSensor`, which computes 23 generic low-level image measurements. Language does not receive names such as color/shape, but this is not yet fully self-organizing raw-pixel feature discovery.

The new family discovery is also unlabeled: APCN is not told that a discovered family is a `color family` or `shape family`. It groups tokens only because their learned relevance vectors are similar.

## Linux UI install

```bash
git pull
chmod +x install_linux.sh
./install_linux.sh
source .venv/bin/activate
python run_desktop_v0_8.py
```

If the server has no graphical session, use X11 forwarding, VNC/RDP, or your cloud provider's desktop. For SSH X11 forwarding:

```bash
ssh -X user@server
cd APCN
source .venv/bin/activate
python run_desktop_v0_8.py
```

If Qt reports an xcb plugin/system-library error on Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libegl1 libxkbcommon-x11-0 libxcb-cursor0 libxcb-xinerama0
```

## Headless-only install

```bash
chmod +x install_headless_linux.sh
./install_headless_linux.sh
source .venv/bin/activate
python train_concepts_v0_8.py --episodes 2400 --eval-samples 400
```

## Desktop controls

The top row controls the procedural teacher. You can choose a color/shape or keep them random, change difficulty, generate a held-out preview, train one episode, run/pause a batch, evaluate, and save/load memory.

The left panel displays the current generated or real image. The dashed rectangle is the teacher's joint-attention signal.

The right panel displays a neuron-like sparse graph. These are **concept/feature nodes, not neural-network neurons**. Larger nodes are more active.

The lower tabs expose readable answers, the activity log, learned concepts, discovered families, and raw debug state.

## Manual teaching

Generate an example, type something such as:

```text
this is a yellow circle
```

then click **Teach current focus**.

For real data click **Load real image**, drag a rectangle around the object, type the teaching utterance, and click **Teach current focus**.

## Command box

```text
train 100
show yellow circle
inspect yellow
test 200
save
load
```

Free-form English in V0.8's command box is not yet a general instruction parser; free-form sentences are used in the teaching utterance field.

## Headless training

```bash
python train_concepts_v0_8.py --episodes 2400 --eval-samples 400
```

Resume from a previous checkpoint:

```bash
python train_concepts_v0_8.py \
  --resume outputs/v0_8/concept_memory_v0_8.json \
  --episodes 2000
```

## Generalization

```bash
python test_generalization_v0_8.py \
  outputs/v0_8/concept_memory_v0_8.json \
  --samples 500 \
  --difficulty 0.92
```

A reference validation run after 2,400 synthetic experiences produced approximately:

- color accuracy: 96.2%
- shape accuracy: 89.2%
- joint accuracy: 85.4%

These are controlled synthetic benchmark numbers, not real-world vision accuracy.

## Concept-family discovery

After enough training, V0.8 compares each token's learned sensory relevance vector. Tokens with similar relevance profiles form connected components. A representative run grouped the color words together and several geometric terms into another family. The learner is not passed the words `color` or `shape` as family labels.

## Tests

```bash
python -m unittest discover -s tests -v
```

V0.8 tests cover visually grounded vs structural role inference, factor disentanglement, automatic family discovery, sparse activation-graph generation, preview isolation, and save/load/resume behavior.

## Next research step

V0.9 should attack the remaining engineered perception bottleneck: replace the fixed 23-dimensional visual front-end with an online self-organizing feature/prototype layer so low-level perceptual primitives themselves can be created and consolidated from raw sensory streams.
