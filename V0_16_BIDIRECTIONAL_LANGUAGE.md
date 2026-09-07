# APCN V0.16 — Bidirectional Semantic Language

V0.16 follows the V0.15 course correction. The objective is not to turn APCN into a text-prediction model. It makes natural-language understanding and generation two directions over the same explicit semantic construction memory.

## Architecture

```text
English input
    ↓
bidirectional construction memory
    ↓
semantic request frame
    ↓
concept / world memory + reasoning
    ↓
semantic answer frame
    ↓
the SAME construction memory
    ↓
English surface realization
```

The semantic responder owns truth. The surface generator receives only a semantic frame and has no reference to the concept store, fact memory, world model, or perception state.

## What this version is trying to prove

1. Multiple learned surface constructions can map to one semantic operation.
2. One explicit semantic frame can be realized in multiple valid surface forms.
3. `parse(generate(S))` preserves `S` for supported semantic frames.
4. Generation cannot independently pull unrelated factual content from world memory.
5. Explicit user teaching in V0.15 memory remains immediately available to V0.16 reasoning.
6. Learned linguistic functions can recombine in syntax that was not a stored TRAIN template.
7. V0.16 adds no visual-training experiences.

## What it is NOT trying to prove

- LLM-level fluency.
- Open-domain English competence.
- Coding ability.
- Internet-scale reading.
- A blind final English benchmark.

The current development material is diagnostic only. Once viewed, it is not treated as blind evaluation material and is not patched phrase-by-phrase.

## Semantic frames

Examples:

```text
ASK_DEFINITION(concept=acceleration)
ASK_DEPENDENCIES(concept=speed)
ASK_KNOWLEDGE(concept=density)
COMPARE(left=speed, right=density)
```

Reasoning produces answer frames such as:

```text
STATE_DEFINITION(
  concept=acceleration,
  definition="velocity change divided by time"
)
```

The construction memory can realize that frame as several surfaces without being allowed to decide what acceleration means.

## Two parsing levels

### Exact learned construction

The strongest path matches a learned slot construction such as:

```text
what is {concept}
what does {concept} depend on
compare {left} and {right}
```

An exact wildcard is not allowed to swallow arbitrary syntax. Captured semantic slots must resolve to explicit known concepts/aliases before the exact path commits.

### Compositional cue recombination

If no trustworthy exact construction matches, V0.16 derives lexical cue → semantic-operation evidence from the TRAIN construction memory itself. It then combines that evidence with explicit concept mentions.

There is no handwritten dictionary such as `depends -> ASK_DEPENDENCIES`.

A conservative generic morphology normalizer is applied equally to TRAIN cues and input, so productive variants such as:

```text
depend  ↔ depends
concept ↔ concepts
dependency ↔ dependencies
learn   ↔ learned
```

can share previously learned linguistic evidence. Irregular or genuinely unseen vocabulary remains unknown until learned.

## Evaluation discipline

The first exact-construction implementation produced these development results:

```text
supported request accuracy       100%
minimum generated variants         5
generation mean variants         5.4
semantic roundtrip exact         100%
unknown honesty                  100%
explicit teaching transfer       100%
generation content firewall      100%
visual experiences changed         0
original DEV parse accuracy        0%
```

The 0% original DEV result was **not hidden or repaired phrase-by-phrase**. The original DEV contains lexical material that is genuinely unseen by TRAIN, so zero-shot semantic knowledge of those words is not assumed.

A second `RECOMBINATION_SPLIT` was frozen before its first execution. Its semantic content words occur in TRAIN, but they are arranged in new syntax. After adding only generic morphology and semantic slot validation—not test-sentence mappings—the current development gate reports:

```text
recombination accuracy           100%
```

This remains development evidence, not a general-English or blind benchmark claim.

## Generation firewall

The production conversation path is:

```text
request frame
    ↓
SemanticResponderV16
    ↓
answer frame containing factual content
    ↓
BidirectionalConstructionMemory.generate(...)
```

The generator itself has no concept/world-memory reference. Architecture tests also pass a fabricated semantic frame with unrelated names and verify that the generated text does not import known domain facts such as acceleration, speed, density, force, pressure, or momentum.

## Explicit teaching priority

Teaching changes semantic memory and therefore has priority over request interpretation. For example:

```text
fluxion means acceleration
```

is routed to the V0.15 explicit semantic teaching path before V0.16 attempts question interpretation. Subsequent V0.16 requests can resolve `fluxion` through that persistent alias without global retraining.

## Research boundary

A fluent future APCN generator may become substantially more sophisticated, but the architectural firewall remains non-negotiable:

> language generation may decide **how to say** a semantic answer, not **what is true**.

If future fluency work requires a compact neural surface realizer, that must be evaluated as a separate hybrid product track rather than silently inserted into the pure non-neural APCN research architecture.
