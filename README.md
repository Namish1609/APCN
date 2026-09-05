# APCN V0.14 — Language-First Live Cognition

Current release candidate: **0.14.0**.

V0.14 keeps the V0.13 persistent world model but changes development priority: most new learning effort now goes into **grounded language construction learning**, while perception remains an active real-world grounding and memory test. The implementation remains non-neural: no backpropagation, gradient descent, trainable neural layers, or pretrained face-embedding model is required.

## V0.14 architecture

```text
                         grounded experience
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
      V0.13 perception/world              V0.14 language-first
               │                                 │
   category + instance memory          entity/relation grounding
   temporal belief / events            + surface constructions
               │                                 │
               │                        ProgramConstructionMemory
               │                                 │
               └───────────────┬─────────────────┘
                               ▼
                    semantic programs + context
                               │
                               ▼
                     persistent cognitive state
```

Default mixed-learning budget:

```text
language / semantic constructions   80%
perception grounding maintenance    20%
```

The ratio is adjustable in the desktop studio.

## Language First

V0.14 adds a bounded sparse `ProgramConstructionMemory`. It learns reusable mappings from surface constructions onto semantic-program schemas while reusing the language-independent semantic primitives already established in earlier releases.

The new procedural language curriculum introduces more varied sentence forms and separate held-out constructions. Testing is read-only: learner episode count is checked before and after held-out evaluation.

The **Language First** tab supports:

- language-only training batches;
- mixed 80/20 language/perception batches;
- held-out construction tests with memory frozen;
- inspection of learned program constructions;
- explicit human paraphrase teaching without global retraining.

This is still controlled semantic-language research, not an open-domain English claim.

## Local desktop camera + self-face experiment

V0.14 adds an opt-in **Self Face Camera** tab for testing the user's own face locally.

It does **not** use a pretrained neural face embedding encoder. Identity comes from deterministic APCN appearance descriptors plus bounded prototype memory. The optional OpenCV Haar cascade is used only to locate a face rectangle; it has no identity role. You can disable automatic location and use a manual face box for the stricter no-pretrained-detector path.

Important boundaries:

- single local enrolled self identity;
- no public-person lookup;
- no demographic inference;
- not security-grade authentication;
- raw camera frames and face crops are not saved;
- bounded compact appearance prototypes only.

A recognition system cannot literally operate on pixels with no representation at all. Here, “no encoder” means **no pretrained learned face embedding network**; APCN still computes a deterministic numerical descriptor from the focused pixels, just as V0.13 did for persistent bottles and other objects.

### Recommended webcam test

```bash
cd ~/APCN
git checkout v0.14-build
git pull
source .venv/bin/activate
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python run_desktop_v0_14.py
```

Open **Self Face Camera**. Start the camera, auto-locate or manually box your face, freeze useful frames and enroll roughly 8–12 varied views. Then test new frames with **Verify: Is This Me?**. A wrong acceptance can be corrected immediately with **This Is NOT Me**, which adds bounded negative evidence rather than retraining a global model.

For a physical webcam, run APCN on the desktop machine that owns the camera. A remote Xvfb/noVNC server usually cannot see the local camera unless the device is explicitly forwarded.

## Existing World Memory

The inherited V0.13 **World Memory** tab remains available for general objects:

```text
focused object pixels
        ↓
fine instance descriptor
        ↓
bounded multi-view memory
        ↓
KNOWN / PROBABLE / AMBIGUOUS / NOVEL
        ↓
VISIBLE / OCCLUDED / OUT_OF_VIEW / LOST
        ↓
trajectory + events + belief-based where()
```

Raw video is not persisted.

## Testing

```bash
python -m unittest tests.test_v0_14 -v
python benchmark_v0_14.py
python benchmark_face_v0_14.py
```

The language benchmark measures held-out semantic constructions and direct paraphrase teaching.

The face benchmark uses **procedural face-like drawings only** to regression-test compact memory, nuisance robustness, unknown rejection and correction. It is not a real-face accuracy result. Real webcam behavior must be measured locally.

The full GitHub Actions workflow preserves the historical V0.7–V0.13 tests, V0.12 hard perception gate, and V0.13 persistent-instance benchmark.

## Scientific boundary

V0.14 is primarily testing whether APCN can move from word/construction cues toward reusable **sentence → semantic-program** structure while retaining grounded persistent world memory. The self-face experiment is a practical stress test of the same bounded instance-memory mechanism, not the central research objective.

It does not claim large-scale language understanding, unrestricted grammar induction, robust real-world face recognition across all conditions, secure biometric authentication, or open-world visual understanding.

Detailed V0.14 notes are in `README_V0_14.md`. Historical release documentation remains in `README_V0_13.md`, `README_V0_12.md`, and earlier version files.
