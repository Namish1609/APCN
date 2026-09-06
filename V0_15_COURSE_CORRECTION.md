# APCN V0.15 course correction

V0.15 is a language-focused milestone, but language is not the knowledge substrate.

## Architectural contract

```text
English input
    ↓
semantic compiler
    ↓
concept / world memory
    ↓
reasoning and explicit operations
    ↓
semantic answer plan
    ↓
English surface realization
```

The concept/world system remains authoritative for acquired knowledge. The language layer may learn sparse statistics that help identify linguistic constructions, but those statistics must not create facts or concepts by themselves.

## Removed from V0.15

- Raw-corpus / n-gram English Exposure subsystem and UI.
- Test-failure-specific lexical bridge templates.
- Learned lexical anchor overrides added after inspecting individual held-out failures.
- Any claim that the current development split is a blind English benchmark.

These experiments were useful diagnostics but are not required for the current proof.

## Retained

- Explicit concept and fact memory.
- Grounded semantic programs.
- Bounded lexical aliases.
- Bounded auxiliary dialogue-act cue statistics.
- Multi-turn semantic dialogue state.
- Explicit unknown / clarification behavior.
- Immediate user teaching and correction.
- V0.14 perception/world state as grounding, with zero new visual-training budget in V0.15.

## Training allocation

V0.15 language-only training uses approximately:

- 35% auxiliary conversational routing.
- 65% grounded semantic-program construction learning.
- 0% new visual training.

The dialogue bootstrap is balanced across training constructions and never uses the teacher's development TEST templates.

## Evaluation protocol

The current teacher TEST split has been inspected during development and is therefore explicitly classified as **development, not blind**. It may be used for regression and diagnosis but not as proof of general English competence.

A future release claim must use a fresh final split that is frozen before evaluation and is not patched phrase-by-phrase after failures are inspected.

## Non-negotiable architecture tests

1. Dialogue statistics must not manufacture world knowledge.
2. Concept knowledge must survive resetting/removing the dialogue router.
3. Unknown concepts remain unknown even after language-only cue training.
4. Explicit teaching updates semantic memory immediately without global retraining.
5. V0.15 training must not alter visual-training state.
6. Raw chat transcripts and raw corpora are not required as long-term memory.

## What V0.15 is trying to prove

Not: "APCN knows all English."

Instead:

> A bounded non-neural language compiler can map multiple linguistic constructions into explicit semantic operations over a separately stored concept/world model, while remaining teachable and honest about unknowns.

That is the milestone. Broader reading and language generation come only after this separation is stable.
