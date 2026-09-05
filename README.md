# APCN V0.13 — Persistent World Memory

Current release: **0.13.0**.

V0.13 extends the V0.12 self-organizing perception system from generic visual concepts into **persistent object instances and temporal world belief**. The implementation remains non-neural: no backpropagation, gradient descent, trainable neural layers, or dense learned weight stack.

## What V0.13 adds

```text
focused pixels / camera frame
        │
        ├── V0.12 category representation
        │      └─ color / shape / generic concepts
        │
        └── V0.13 fine instance representation
               └─ local texture / chromatic layout / edges / occupancy
                        │
                        ▼
             bounded multi-view instance memory
                        │
          appearance + category + temporal continuity
                        │
        KNOWN / PROBABLE / AMBIGUOUS / NOVEL
                        │
                        ▼
              PersistentWorldModel
              ├─ VISIBLE
              ├─ OCCLUDED
              ├─ OUT_OF_VIEW
              └─ LOST
                        │
                        ▼
        trajectories + events + belief-based `where`
```

A named object such as `my_bottle` stores only bounded aggregate appearance prototypes. Raw camera/video frames are working memory and are not written into the APCN checkpoint.

Default instance-memory bound:

```text
positive appearance modes <= 8 / instance
negative correction modes <= 4 / instance
raw images retained       = 0
raw video retained        = 0
```

V0.13 also supports immediate human identity correction. A correction adds compact negative evidence to the wrong instance and positive evidence to the correct instance; it does not retrain a global model.

## Desktop camera and World Memory

```bash
cd ~/APCN
git checkout main
git pull
source .venv/bin/activate
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python run_desktop_v0_13.py
```

Open **World Memory**. You can load an image or start camera device `0`, freeze a frame, define the normalized focus box, teach a persistent name such as `my_bottle`, observe later views, correct a wrong identity, mark absence/occlusion, and ask **Where is it?**.

The camera currently uses a manually specified focus box. This is intentional: V0.13 tests persistent identity/world memory separately from unrestricted open-world detection.

## Recognition semantics

V0.13 uses open-set evidence rather than a single nearest-neighbour threshold.

- **KNOWN** — strong absolute appearance evidence and separation from competitors.
- **PROBABLE** — enough combined appearance/margin evidence to commit identity with uncertainty.
- **AMBIGUOUS** — candidate evidence exists, but identity is deliberately non-committing and cannot silently update a track.
- **NOVEL** — insufficient evidence for an existing instance.

Temporal belief is separate from object identity. A known instance can be `VISIBLE`, `OCCLUDED`, `OUT_OF_VIEW`, or `LOST` while retaining its persistent identity memory.

## Testing

Run the V0.13 unit tests and controlled persistent-instance benchmark:

```bash
python -m unittest tests.test_v0_13 -v
python benchmark_v0_13.py
```

The benchmark teaches two visually similar instances sharing the same coarse category and evaluates held-out multi-view identity, similar-instance disambiguation, unknown rejection, occlusion, reappearance, one-step correction, and bounded memory. Ground-truth identity is used only by the evaluator and is not passed to read-only matching.

The full GitHub Actions suite also preserves the historical V0.7–V0.12 tests and V0.12 hard perception gate.

## Scientific boundary

V0.13 does **not** claim solved open-world object detection, unrestricted video understanding, human-action recognition, or biometric face identification. It establishes the narrower mechanism needed by HUD: `this observed object now = that persistent object seen earlier`, with explicit uncertainty, bounded memory, and no full-model retraining.

Historical release documentation remains available in `README_V0_12.md`, `README_V0_11.md`, and earlier version files.
