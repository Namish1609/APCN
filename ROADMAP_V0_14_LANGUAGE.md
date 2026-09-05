# APCN V0.14 — Language-First Research Roadmap

## Decision

After V0.13 closes the persistent object/world-memory milestone, APCN should shift the majority of research effort from visual feature engineering to grounded semantic language learning.

Target allocation for the next milestone:

```text
language / semantic learning      ~70-80%
perception + camera maintenance   ~20-30%
```

The reason is architectural, not cosmetic. V0.12/V0.13 now provide enough perception/world-state machinery to generate grounded experiences. The larger unresolved scaling question is whether APCN can acquire increasingly abstract concepts and executable semantic structure from language without turning every new concept into a hand-coded rule.

## What V0.13 is sufficient for

V0.13 should remain the visual/world-memory baseline:

- generic visual concept grounding;
- fine persistent object-instance appearance;
- bounded multi-view instance memory;
- explicit uncertainty/open-set recognition;
- VISIBLE/OCCLUDED/OUT_OF_VIEW/LOST belief;
- camera/image input with manually focused object region;
- immediate correction;
- no raw frame archive.

Do not make V0.14 depend on solving unrestricted open-world vision first.

## V0.14 research question

Can APCN learn the mapping

```text
utterance + discourse + grounded world state + action/outcome + feedback
                              ↓
                 reusable semantic program
```

while retaining bounded/sparse memory and generalizing to novel combinations?

## Core architecture

```text
WORLD MODEL / EXPERIENCE
    ├─ persistent instances
    ├─ properties
    ├─ relations
    ├─ events
    ├─ actions/outcomes
    └─ discourse focus
             │
             ▼
SURFACE LANGUAGE
             │
             ▼
lexical hypotheses
             │
             ▼
role / construction hypotheses
             │
             ▼
REFERENCE RESOLUTION
             │
             ▼
SEMANTIC PROGRAM
    ├─ ASSERT
    ├─ QUERY
    ├─ GOAL
    ├─ RELATION
    ├─ GROUP
    ├─ SEQUENCE
    ├─ NEGATION
    └─ REFERENCE
             │
             ▼
WORLD-MODEL EXECUTION / ANSWER / ACTION
             │
             ▼
feedback + correction + consolidation
```

The internal operator inventory may remain engineered initially. The scientific test is whether surface cues, lexical mappings, argument roles, constructions and reusable compositions are learned from experience rather than hardwired to English words.

## Curriculum

### Stage A — grounded reference

Teach language directly against V0.13 world instances:

```text
this is the bottle
this is my bottle
where is the bottle?
where is my bottle?
it moved
where is it?
```

Required tests:

- same category, different persistent instances;
- pronoun/reference continuity;
- ambiguous reference must remain uncertain;
- correction updates the interpretation immediately.

### Stage B — relations and roles

Use scenes and transitions to learn:

```text
inside
left of
right of
above
below
near
moves
pushes
contains
```

The system should infer who is actor/source/target from repeated grounded outcomes rather than memorizing one sentence.

### Stage C — intent

Contrast the same content under different communicative functions:

```text
the bottle is in the box        -> ASSERT
is the bottle in the box?       -> QUERY
put the bottle in the box       -> GOAL
```

Intent should be learned from demonstrations/feedback and abstract constructions, not fixed English sentence lookup.

### Stage D — operators and composition

Learn reusable function-word/construction behavior:

```text
and   -> GROUP
then  -> SEQUENCE
not   -> NEGATION
```

Use arbitrary aliases in anti-cheating tests, for example learning `ka` as a conjunction from demonstrations.

### Stage E — definitions / concept-from-concept learning

Once component concepts are grounded, allow statements such as:

```text
speed means distance divided by time
density means mass divided by volume
```

to build executable semantic definitions.

Every derived concept must carry a grounding audit:

- known dependencies;
- unresolved dependencies;
- executable definition if available;
- examples/counterexamples;
- confidence/evidence.

## Benchmarks

V0.14 should not be released on training accuracy alone. It should include:

1. **arbitrary vocabulary** — replace English content words/operators with invented tokens;
2. **held-out composition** — learn constituents separately, test unseen combinations;
3. **reference** — resolve `it`, `that one`, names and category descriptions across turns;
4. **paraphrase** — multiple surface forms map to one semantic program;
5. **polysemy** — one token can map to multiple semantic hypotheses depending on context;
6. **correction** — recent explicit correction must overcome stale evidence;
7. **definition chaining** — derived concepts execute through dependency graphs;
8. **unknown dependency detection** — never pretend a definition is grounded when prerequisites are missing;
9. **bounded memory** — no raw sentence archive required for ordinary learning;
10. **growth test** — measure memory/search cost while scaling concepts and constructions.

## Scaling experiment

Use a progression such as:

```text
100 primitive concepts
500 derived concepts
2,000 constructions
10,000 held-out combinations
```

Measure:

- semantic exact accuracy;
- intent accuracy;
- reference accuracy;
- definition execution accuracy;
- correction latency;
- memory bytes / concept;
- inference/search time;
- duplicate concept creation;
- forgetting/interference.

## Perception policy during V0.14

Perception remains active but should be treated mainly as a grounding interface and regression surface. Improve it only when a language experiment is blocked by a clearly measured perceptual limitation.

Do not spend the next milestone chasing unrestricted visual recognition before testing the central APCN scaling hypothesis: **can grounded concepts compose into increasingly abstract language/science knowledge while the cognitive engine stays approximately fixed?**

## Human-camera boundary

The V0.13 camera path is for objects and anonymous world-state testing. V0.14 should not add biometric face identification or persistent named-person recognition from facial appearance. Anonymous person/face presence detection or short-term non-identifying tracking may be considered separately if needed for grounding actions.

## Proposed release order

```text
V0.13  persistent object/world memory + camera I/O
V0.14  grounded semantic language + reference + composition
V0.15  concept-from-concept definitions + executable science/math
V0.16  autonomous study / unresolved-dependency-driven learning
V0.17  large concept-memory scaling experiment
```
