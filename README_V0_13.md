# APCN V0.13 — Persistent World Memory

V0.13 extends the V0.12 self-organizing visual representation into persistent object instances and an explicit temporal world belief.

## Research question

Can APCN learn a named object from a small number of observations, keep only bounded compact appearance evidence, re-identify the same instance later, represent uncertainty/occlusion explicitly, survive restart, and accept an immediate correction without gradient training?

## Architecture

```text
image / camera frame
    ↓
focused object region (manual box in V0.13)
    ├─────────────────────────────────────┐
    ↓                                     ↓
V0.12 category representation      V0.13 fine instance representation
(color / shape / generic concept)   (texture / chromatic layout / edges / occupancy)
    │                                     │
    └──────────────────┬──────────────────┘
                       ↓
              bounded multi-view instance memory
                       ↓
      appearance evidence + category + temporal continuity
                       ↓
          KNOWN / PROBABLE / AMBIGUOUS / NOVEL
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

V0.13 open-set matching uses both absolute similarity and best-vs-second identity margin. `AMBIGUOUS` is intentionally non-committing: ambiguous evidence does not silently update a persistent identity or auto-create a new object.

## Temporal belief

`PersistentWorldModel` tracks each instance separately from category recognition. A track has:

- normalized position and bounding box;
- velocity estimate;
- confidence;
- last-seen frame/time;
- bounded trajectory history;
- explicit state: `VISIBLE`, `OCCLUDED`, `OUT_OF_VIEW`, or `LOST`.

The model never answers a location query from simulator truth. `where(name)` uses only its own tracked belief.

## World Memory UI + desktop camera

Run:

```bash
python run_desktop_v0_13.py
```

Open **World Memory**.

For an image:

1. Click **Load Image / Frame**.
2. Set the normalized focus box around the object.
3. Enter a persistent object name such as `my_bottle`.
4. Click **Teach Named Instance** from several different views.
5. Load a later frame and click **Observe / Match (no learning label)**.

For a local desktop camera:

1. Click **Start Camera**.
2. Adjust the normalized focus box around the object you want APCN to observe.
3. Click **Freeze Frame** when you want a stable teaching/testing observation.
4. Use **Teach Named Instance** or **Observe / Match** on that frozen frame.
5. Restart the camera for another viewpoint and repeat.
6. Click **Stop Camera** when finished.

Camera frames are working memory only. They are not written to the APCN checkpoint.

Camera device `0` is used by default. On a local desktop, the OS may ask for camera permission. A remote Xvfb/noVNC machine normally has no physical camera unless a device is explicitly forwarded.

If APCN chose the wrong object identity, enter the correct object name and click **Correct Current Observation To Name**. Use **No Detection This Frame** with/without the occluder checkbox to update world belief. Use **Where is it?** to inspect APCN's current belief.

The manual box is an intentional boundary for V0.13: this version evaluates persistent object identity/world memory separately from general open-world object detection.

## Human-face boundary

The camera path is intended for object/world-memory research and anonymous visual tracking. V0.13 does not implement biometric face identification or persistent named-person recognition from facial appearance.

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

V0.13 does **not** solve open-world detection, arbitrary real-world category recognition, biometric identification, human pose/action recognition, or unrestricted video understanding. Real-image/camera testing currently uses a manually specified focus box. The point of V0.13 is to test persistent instance memory and temporal belief once an object observation is available.

A successful V0.13 is evidence toward the HUD use case: `this object now = that object seen earlier`, with explicit uncertainty and persistent memory, without full-model retraining.
