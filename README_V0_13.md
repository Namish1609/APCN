# APCN V0.13 — Persistent World Memory

V0.13 extends the V0.12 self-organizing visual representation into persistent object instances and an explicit temporal world belief.

## Research question

Can APCN learn a named object from a small number of observations, keep only bounded compact appearance evidence, re-identify the same instance later, represent uncertainty/occlusion explicitly, survive restart, and accept an immediate correction without gradient training?

## Architecture

```text
image / frame
    ↓
focused object region (manual or upstream detector)
    ↓
V0.12 self-organizing visual descriptor
    ↓
bounded multi-view instance memory
    ↓
appearance evidence + optional category + temporal continuity
    ↓
KNOWN / PROBABLE / AMBIGUOUS / NOVEL identity
    ↓
PersistentWorldModel
    ├─ VISIBLE
    ├─ OCCLUDED
    ├─ OUT_OF_VIEW
    └─ LOST
    ↓
trajectory + event history + belief-based location answer
```

The current implementation remains non-neural: no backpropagation, gradient descent, trainable neural layers or dense learned weight stack.

## Persistent instance memory

A named instance such as `my_bottle` keeps a bounded positive appearance bank and a smaller negative/correction bank. Each entry is an aggregate streaming prototype (`count`, `mean`, `m2`), not a stored image.

Default bound:

```text
positive appearance modes: <= 8 / instance
negative correction modes: <= 4 / instance
raw images retained: 0
raw video frames retained: 0
```

A human correction immediately adds negative evidence to the wrong identity and positive evidence to the correct identity.

## Temporal belief

`PersistentWorldModel` tracks each instance separately from category recognition. A track has:

- normalized position and bounding box;
- velocity estimate;
- confidence;
- last-seen frame/time;
- bounded trajectory history;
- explicit state: `VISIBLE`, `OCCLUDED`, `OUT_OF_VIEW`, or `LOST`.

The model never answers a location query from simulator truth. `where(name)` uses only its own tracked belief.

## World Memory UI

Run:

```bash
python run_desktop_v0_13.py
```

Open **World Memory**.

1. Load a real image/frame.
2. Set the normalized focus box around the object.
3. Enter a persistent name such as `my_bottle`.
4. Click **Teach Named Instance** from several different views.
5. Load a later frame and click **Observe / Match (no learning label)**.
6. If APCN chose the wrong identity, enter the correct name and click **Correct Current Observation To Name**.
7. Use **No Detection This Frame** with/without the occluder checkbox to update world belief.
8. Ask **Where is it?** to inspect APCN's belief state.

The manual box is an intentional boundary for V0.13: this version evaluates persistent identity/world memory separately from general open-world object detection.

## Migration

On first launch, V0.13 loads `outputs/v0_13` if present. Otherwise, if `outputs/v0_12/session_v0_12.json` exists, it imports V0.12 perception, language, definitions, discourse, error memory and consolidation history. V0.13 world/instance memory begins clean.

## Controlled benchmark

```bash
python benchmark_v0_13.py
```

The benchmark teaches two visually similar instances with the same coarse category/color/shape but different ordinary microtexture. It then tests:

- held-out multi-view identity;
- similar-instance disambiguation;
- unknown-instance rejection;
- occlusion belief;
- reappearance identity recovery;
- immediate correction;
- bounded memory / no frame archive.

The synthetic teacher's ground-truth instance name is used only by the evaluator. It is not supplied to read-only identity matching.

## Scientific boundary

V0.13 does **not** solve open-world detection, arbitrary real-world category recognition, human pose/action recognition, or unrestricted video understanding. Real-image testing currently uses a manually specified focus box. The point of V0.13 is to test persistent instance memory and temporal belief once an object observation is available.

A successful V0.13 is evidence toward the HUD use case: `this object now = that object seen earlier`, with explicit uncertainty and persistent memory, without full-model retraining.
