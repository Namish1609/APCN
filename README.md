# APCN V0.11 — Unified Concept Memory + Self-Consolidation

Current release: **0.11.0**.

APCN V0.11 moves the project from manually repeating large training batches toward an explicit self-diagnosis and consolidation loop:

```text
read-only test
    ↓
aggregate recurring errors
    ↓
rank weak concept boundaries / language constructions
    ↓
generate targeted contrast experiences
    ↓
consolidate compact memory
    ↓
read-only retest
```

It remains non-neural in the current implementation: no backpropagation, gradient descent, trainable neural layers or dense learned weight stack.

## V0.11 highlights

- bounded multi-prototype visual concepts instead of a single centroid per word;
- aggregate error memory instead of retaining every failure;
- automatic non-gradient consolidation prescriptions;
- learned sentence-construction induction;
- persistent discourse entity registry for `it`, `that object`, and stable instance identity;
- stricter identity-aware reference testing;
- unified sparse concept graph linking perception, language and definitions;
- idempotent graph synchronization—viewing the graph cannot change knowledge;
- V0.10 compact-memory migration without replaying old images or sentences;
- persistent error/discourse/consolidation state across restarts;
- new **Consolidation** desktop page with priorities, memory audit, concept graph and before→after curves.

## Update and run the desktop studio

```bash
cd ~/APCN
git checkout main
git pull
source .venv/bin/activate
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python run_desktop_v0_11.py
```

The UI remains designed for a 1366×768 desktop and retains the Perception, Language, Definitions, Ask APCN and Testing Ground pages from V0.10.2.

## Continue from your V0.10 training

If you already trained V0.10, open the **Consolidation** tab and click:

```text
Migrate V0.10 Memory
```

The normal migration paths are:

```text
outputs/v0_10/perception/concept_memory_v0_8.json
outputs/v0_10/language_memory_v0_10.json
outputs/v0_10/concept_store_v0_10.json
```

V0.11 converts those compact memories; it does not need the original thousands of images or sentences.

Then click:

```text
Run 1 Automatic Consolidation Cycle
```

The cycle performs diagnosis → targeted visual/language learning → retesting and shows the before/after curves.

## Headless consolidation

```bash
python train_v0_11.py \
  --visual-memory outputs/v0_10/perception/concept_memory_v0_8.json \
  --language-memory outputs/v0_10/language_memory_v0_10.json \
  --concept-memory outputs/v0_10/concept_store_v0_10.json \
  --visual 0 \
  --language 0 \
  --consolidation-cycles 3 \
  --definitions
```

Or start a clean V0.11 experiment:

```bash
python train_v0_11.py \
  --visual 2000 \
  --language 3200 \
  --visual-test 500 \
  --language-test 600 \
  --definitions \
  --consolidation-cycles 2
```

## What is stored?

The visual learner does **not** retain all training images. Long-term visual memory is compact sufficient statistics plus a bounded prototype bank. The language learner does **not** retain all training sentences; it retains aggregate cue/ngram/semantic counts and induced construction statistics. Error memory collapses repeated mistakes into bounded signatures and the discourse registry is bounded working memory.

## Tests

```bash
python -m unittest discover -s tests -v
```

See [`README_V0_11.md`](README_V0_11.md) for the full V0.11 architecture, migration workflow and scientific boundaries.

## Current scientific boundary

V0.11 does **not** establish that APCN scales to general intelligence or replaces transformers. The visual front end still uses the engineered anonymous 23-dimensional sensor. The V0.11 experiment is whether compact explicit knowledge can diagnose and target its own weaknesses without gradients or an unbounded episode archive.

A likely next major research direction is self-organizing perception from more generic local pixel structure rather than indefinitely improving the handcrafted sensor.
