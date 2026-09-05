# APCN V0.14 — Language-First Live Cognition

V0.14 changes the development priority rather than replacing the V0.13 world model. The system remains non-neural: no gradient descent, backpropagation, trainable neural layers, or pretrained face-embedding model is required by the new self-face experiment.

## Why language-first now

V0.12 and V0.13 established a workable perception path for controlled concepts, persistent object instances, temporal identity, occlusion, reappearance and compact correction. The larger remaining architectural risk is whether grounded language can scale beyond a small set of fixed sentence templates.

V0.14 therefore uses an approximate default learning budget of:

```text
language / semantic construction learning   80%
perception maintenance / grounding          20%
```

The ratio is adjustable in the desktop studio.

## New language mechanism

V0.14 adds `ProgramConstructionMemory`, a bounded sparse mapping between learned surface constructions and semantic-program schemas.

```text
utterance
   │
   ├─ learned entity spans
   ├─ learned relation spans
   └─ surface construction
           │
           ▼
  numbered abstract pattern
  <E0> ... <R0> ... <E1>
           │
           ▼
 learned program schema
 ASSERT / QUERY / GOAL / GROUP / SEQUENCE / NEGATE
           │
           ▼
 bind current grounded entities + relations
           │
           ▼
 executable semantic program
```

The learner keeps the older lexical and construction memories and adds this higher-level program-construction layer. Memory is bounded by a maximum pattern count rather than retaining every sentence.

The procedural teacher includes richer training surface forms and separate held-out constructions. Read-only held-out tests verify that the learner is not changing memory while being evaluated.

## Explicit human paraphrase teaching

The Language First tab can generate a grounded semantic demonstration. You can type another sentence that should mean the same thing and teach that paraphrase directly, without global retraining.

This is still a constrained semantic-program experiment. V0.14 does not claim open-domain English understanding.

## Local self-face camera experiment

V0.14 adds an opt-in **Self Face Camera** tab for the user's own local desktop test.

Important boundary:

- it is single-identity local self-verification;
- it does not search for or identify public people;
- it does not infer demographic attributes;
- it is not security-grade authentication;
- raw camera frames and face crops are not saved;
- identity uses bounded APCN appearance prototypes;
- no pretrained neural face embedding encoder is used.

The optional `ClassicalFaceLocator` uses OpenCV's Haar cascade only to locate a face rectangle. It has no identity role. If you want a stricter no-pretrained-detector experiment, disable automatic location and set the face box manually.

A recognition system still needs a numerical representation of pixels. V0.14 uses deterministic handcrafted appearance descriptors, not a learned face embedding network.

### Recommended desktop test

```bash
cd ~/APCN
git checkout v0.14-build
git pull
source .venv/bin/activate
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python run_desktop_v0_14.py
```

Open **Self Face Camera**:

1. Start Camera.
2. Use Auto Locate Face or set the normalized face box manually.
3. Freeze a useful frame.
4. Press **Enroll Current Face View**.
5. Repeat for roughly 8–12 varied views: slight head angle, distance, expression and lighting.
6. On new frames press **Verify: Is This Me?**.
7. If a non-self frame is incorrectly accepted, press **Correction: This Is NOT Me** to add compact negative evidence.

A remote Xvfb/noVNC server normally cannot access your physical desktop webcam unless the device is forwarded. For direct webcam testing, run the application on the desktop machine that owns the camera.

## Existing V0.13 world memory remains available

The inherited **World Memory** tab still supports:

- camera or image frames;
- persistent named objects such as `my_bottle`;
- bounded multi-view identity;
- `KNOWN / PROBABLE / AMBIGUOUS / NOVEL` identity evidence;
- `VISIBLE / OCCLUDED / OUT_OF_VIEW / LOST` temporal belief;
- immediate human correction;
- belief-based `where` queries.

The face experiment is intentionally separate from generic object memory so that self-face behavior cannot silently change the object-tracking benchmark.

## Tests

```bash
python -m unittest tests.test_v0_14 -v
python benchmark_v0_14.py
python benchmark_face_v0_14.py
```

`benchmark_v0_14.py` measures held-out language constructions and explicit paraphrase learning.

`benchmark_face_v0_14.py` uses procedural face-like drawings only to regression-test invariance, correction, bounded memory and raw-frame retention. Its scores are **not** real-face accuracy claims; actual webcam performance must be measured locally by the user.

## Scientific boundary

V0.14 is testing two hypotheses:

1. language structure can increasingly be stored as sparse reusable semantic constructions rather than fixed parser rules or a dense neural language model;
2. persistent self-identity can be tested with compact non-neural appearance memory without retaining raw video.

It does not establish large-scale language understanding, unconstrained grammar induction, robust face recognition across the full range of real-world pose/lighting, or secure biometric authentication.
