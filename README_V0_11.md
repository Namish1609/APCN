# APCN V0.11 — Unified Concept Memory + Consolidation

V0.11 is the first version aimed at getting APCN out of the manual "train a large batch, inspect failures, train another large batch" loop.

## Core changes

### 1. Unified sparse concept graph
Perception, language and definition knowledge are synchronized into one explicit graph. Subsystems keep their own evidence, while lexical bridges can create evidence-backed `SAME_CONCEPT_HYPOTHESIS` edges.

### 2. Aggregate error memory
Repeated failures are collapsed into signatures such as:

```text
visual_shape: rectangle -> ellipse
language_program/reference: GOAL -> ASSERT
```

The system does **not** retain every failed image or sentence. Only aggregate counts/recency plus a bounded number of representative strings are kept.

### 3. Non-gradient consolidation
APCN ranks weak concept boundaries and constructions, then generates targeted contrastive experiences. This is not backpropagation. The diagnostic objective combines prediction error, concept ambiguity and memory complexity.

### 4. Bounded multi-prototype visual concepts
A visual word can keep up to a small fixed number of streaming prototypes rather than collapsing all appearances into one centroid. Prototypes are aggregate clusters, not retained training images.

### 5. Learned construction induction
V0.11 abstracts grounded content spans to forms such as:

```text
it is the case that the <ENTITY> is <REL> the <ENTITY>
```

and learns recurring construction -> intent evidence. Intent phrases are not hardcoded to `ASSERT`, `QUERY` or `GOAL` in this module.

### 6. Memory migration
Existing compact V0.8/V0.10 memories can be loaded into the V0.11 learners. Old experience does not need to be replayed from raw data.

## Memory model

Visual training stores global and per-token sufficient statistics plus a bounded prototype bank. Raw image count retained by the learner: **0**.

Language training stores cue/feature/position counts plus learned construction counts. Raw sentence count retained by the learner: **0**.

A bounded error-memory layer keeps repeated confusion signatures instead of an unbounded archive.

## Headless experiment

```bash
python train_v0_11.py \
  --visual 2000 \
  --language 3200 \
  --visual-test 500 \
  --language-test 600 \
  --definitions
```

Outputs are written under `outputs/v0_11/`.

## Scientific boundary

V0.11 is still not an LLM replacement. The perceptual front end still uses the engineered anonymous 23-dimensional sensor. The important V0.11 experiment is whether explicit knowledge can consolidate, migrate, and target its own weak boundaries without gradient training or storing every episode.
