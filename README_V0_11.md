# APCN V0.11 — Unified Concept Memory + Self-Consolidation

V0.11 is the first APCN version aimed at escaping the manual loop:

```text
train a huge batch -> inspect failures -> manually decide what to teach -> repeat
```

The new loop is:

```text
read-only test
    -> aggregate recurring errors
    -> rank weak concept boundaries / constructions
    -> generate targeted contrast experiences
    -> consolidate compact memory
    -> read-only retest
```

No backpropagation or gradient descent is used.

## Core changes

### 1. Unified sparse concept graph
Perception, language and definition knowledge are synchronized into one explicit graph. Subsystems retain their own evidence while lexical bridges can create evidence-backed `SAME_CONCEPT_HYPOTHESIS` edges.

Graph synchronization is **idempotent**. Refreshing the UI rebuilds the graph from current compact memories; viewing it cannot strengthen knowledge.

### 2. Aggregate error memory
Repeated failures are collapsed into signatures such as:

```text
visual_shape: rectangle -> ellipse
language_program/reference: GOAL -> ASSERT
```

The system does not retain every failed image or sentence. It stores aggregate count/recency and a bounded diagnostic sample of representative strings.

### 3. Non-gradient consolidation
APCN ranks weak concept boundaries and language constructions and generates targeted evidence. The diagnostic objective combines:

- prediction error,
- concept ambiguity,
- sparse-memory complexity.

The objective is measured; it is never differentiated.

### 4. Bounded multi-prototype visual concepts
A visual word can keep a small fixed number of streaming prototypes instead of collapsing all appearances into one centroid. This helps distinguish nuisance modes such as wide/tall/rotated rectangles from ellipses while still retaining **zero raw training images**.

Default maximum: 6 aggregate prototypes per token.

### 5. Learned construction induction
V0.11 abstracts grounded content spans into structures such as:

```text
it is the case that the <ENTITY> is <REL> the <ENTITY>
```

and learns construction -> intent evidence. The construction inducer does not contain a literal map such as `"it is the case" -> ASSERT`.

### 6. Persistent discourse entity registry
Reference is no longer modeled only as a single `current_focus` variable. The discourse layer maintains a small working registry containing:

- stable entity instance IDs,
- recency,
- salience,
- last grammatical/semantic role,
- current focus.

Reference testing is stricter in V0.11: the benchmark first asks APCN to interpret the previous sentence, builds context from APCN's own prediction, and then tests `it / that object / the same object`. Teacher entity IDs are not injected into the registry.

### 7. Memory migration
Existing V0.8/V0.10 compact memories can be loaded into V0.11. Old images and sentences do not need to be replayed.

The desktop app includes **Migrate V0.10 Memory** and checks the normal V0.10 output paths.

## What is actually stored?

### Visual
For long-term grounding the learner stores:

- global count/sum/sum-of-squares,
- per-token count/sum/sum-of-squares,
- a bounded prototype bank.

Raw training images retained by the learner: **0**.

### Language
The learner stores:

- cue/ngram counts,
- cue -> semantic-feature counts,
- positional cue counts,
- induced construction counts.

Raw training sentences retained by the learner: **0**.

### Error memory
Repeated errors are collapsed into signatures. Representative strings are bounded.

### Discourse
Discourse entities are working memory, not an unbounded conversation transcript.

## Desktop studio

```bash
python run_desktop_v0_11.py
```

V0.11 keeps the Perception, Language, Definitions, Ask APCN and Testing Ground pages and adds a **Consolidation** page with:

- learning-priority table,
- automatic consolidation cycle,
- live percentage/stage progress,
- before -> after performance graph,
- unified concept graph visualization,
- memory + discourse audit,
- V0.10 memory migration.

Do not run ordinary Perception/Language training at the same time as a consolidation worker; the UI prevents concurrent writers.

## Headless experiment

```bash
python train_v0_11.py \
  --visual 2000 \
  --language 3200 \
  --visual-test 500 \
  --language-test 600 \
  --definitions \
  --consolidation-cycles 2
```

### Continue from V0.10 memory

```bash
python train_v0_11.py \
  --visual-memory outputs/v0_10/perception/concept_memory_v0_8.json \
  --language-memory outputs/v0_10/language_memory_v0_10.json \
  --concept-memory outputs/v0_10/concept_store_v0_10.json \
  --visual 500 \
  --language 1000 \
  --consolidation-cycles 3
```

Outputs are written under `outputs/v0_11/`.

## Tests

```bash
python -m unittest discover -s tests -v
```

V0.11 regressions include bounded prototype memory, aggregate error memory, construction induction, graph idempotence, stable discourse instance identities, unambiguous reference-teacher targets, unified graph bridges and UI consolidation controls.

## Scientific boundary

V0.11 is still not an LLM replacement, and the perceptual front end still uses the engineered anonymous 23-dimensional sensor. The important V0.11 experiment is narrower:

> Can explicit knowledge migrate, consolidate, target its own weak boundaries, and improve from compact error evidence without gradient training or storing every episode?

The next major research target after this should be self-organizing perception from more generic local pixel structure rather than indefinitely refining the handcrafted 23-dimensional front end.
