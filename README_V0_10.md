# APCN V0.10 — Automatic Learning + Concept-from-Concept Definitions

V0.10 corrects the main V0.9 usability/training flaws and implements the next research milestone:

- **V0.9 goal retained:** grounded semantic language acquisition
- **V0.10 added:** definitions and concept-from-concept learning

The V0.10 desktop is designed for a 1366×768 desktop and keeps four focused tabs: Perception, Language, Definitions, Testing Ground.

## Update and run

```bash
cd ~/APCN
git pull
source .venv/bin/activate
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python run_desktop_v0_10.py
```

## 1. Perception training progress is now actually live

The previous UI could appear to jump from 0 to the requested batch size. V0.10 processes GUI training through a paced timer and displays:

```text
completed / requested
remaining
percentage
current curriculum phase
total visual episodes
current generated example
current APCN prediction
```

Pause preserves the unfinished batch; the same button becomes Resume.

Large batches automatically process more experiences per UI tick so 500, 2,000 and 10,000-example runs remain practical while still visibly progressing.

## 2. Language training is fully automatic

There is no user-facing selector for relation / intent / group / sequence / negation / reference.

The automatic curriculum internally tracks:

```text
grounding
intent
group
sequence
negation
reference
```

Each skill keeps evidence count and a recent-competence estimate. Prerequisites unlock automatically. Within the unlocked set, APCN prioritizes weaker skills but keeps exposure approximately balanced so one difficult skill cannot dominate the whole corpus.

This balance is important because cross-situational learning needs negative/contrastive evidence. V0.9 could over-train one failed operator and accidentally make ordinary words correlate with that operator.

## 3. Much larger procedural language space

V0.10 does not store a tiny fixed sentence list. `RichSemanticTeacher` composes utterances from:

- 6 colors
- 5 shapes
- 6 spatial relations
- multiple relation aliases (`inside/in/within`, `above/over`, etc.)
- many assertion/question/command constructions
- grouping constructions
- sequence connectors
- negation cues
- contextual reference constructions

Together with random entity combinations, the teacher can generate a very large number of distinct grounded utterances. Test templates are separated from training templates.

The learner receives the utterance paired with a language-independent semantic program produced by the teacher/world. It never receives the teacher's word maps.

## 4. Generated language testing

The Testing Ground has one generated semantic test. You do not choose what linguistic feature to test.

It automatically balances across all curriculum skills and uses held-out wording. Testing is read-only and verifies that language memory did not change.

The testing layout intentionally has **no standalone score cards**:

```text
Intent confusion matrix | Relation confusion matrix
Failures                | Graphs
```

Every test adds a graph point at the current number of learned language experiences. The graph tracks exact-program, intent, relation and operator correctness over time.

## 5. V0.10 concept-from-concept learning

V0.10 adds an explicit `ConceptStore` and generic definition constructions.

Examples:

```text
speed is distance divided by time
density is mass divided by volume
momentum is the product of mass and velocity
acceleration is velocity change divided by time
force is the product of mass and acceleration
power is work divided by time
pressure is force divided by area
```

These become dependency/expression graphs rather than flat text strings.

For example:

```text
FORCE
└── MUL
    ├── MASS
    └── ACCELERATION
        └── DIV
            ├── VELOCITY CHANGE
            └── TIME
```

With numeric primitive values, executable definitions can be evaluated:

```text
mass = 2
velocity change = 20
time = 4

acceleration = 20 / 4 = 5
force = 2 * 5 = 10
```

The definition parser is not keyed to literal science vocabulary. The same construction works with arbitrary names:

```text
zorp is dax divided by mip
```

if `dax` and `mip` are existing concepts.

## 6. Grounding audit

V0.10 refuses to equate text connectivity with complete understanding.

If it learns:

```text
mystery momentum is the product of mass and ghost velocity
```

and `ghost velocity` is unknown, the concept remains explicitly incomplete:

```text
known: true
complete: false
unresolved: [ghost velocity]
```

This is the intended bridge toward textbook/science learning: definitions can create hypotheses, while missing dependencies remain visible instead of being silently treated as understood.

## Scientific boundary

V0.10 does **not** yet prove open-domain English or scientific understanding.

- perception still uses the V0.8.2 engineered 23-dimensional visual front end;
- semantic programs use a small engineered set of internal primitives;
- definition constructions are currently a compact generic grammar, not unrestricted textbook parsing;
- primitive scientific quantities such as mass and time are bootstrap placeholders that must ultimately be grounded through measurement/experience.

What V0.10 tests is whether:

1. grounded language acquisition can be made automatic and adaptive;
2. held-out generated language performance improves with experience;
3. new concepts can be constructed from already-known concepts;
4. dependency gaps can be detected explicitly;
5. derived concepts can be executable rather than just textual labels.

## Headless language + definition training

```bash
python train_v0_10.py \
  --language-experiences 2800 \
  --test-samples 600 \
  --definitions
```

## Tests

```bash
python -m unittest discover -s tests -v
```

V0.10-specific tests cover automatic skill selection, read-only generated evaluation, semantic improvement, arbitrary definition composition, multi-level derived concepts, unresolved dependency auditing, cycle rejection, persistence, and the built-in science-definition bootstrap.
