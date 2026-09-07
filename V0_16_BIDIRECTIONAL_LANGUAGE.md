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
6. V0.16 adds no visual-training experiences.

## What it is NOT trying to prove

- LLM-level fluency.
- Open-domain English competence.
- Coding ability.
- Internet-scale reading.
- A blind final English benchmark.

The current DEV constructions are diagnostic only. Once viewed, they are not treated as blind evaluation material and are not patched phrase-by-phrase.

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

## Research boundary

A fluent future APCN generator may become substantially more sophisticated, but the architectural firewall remains non-negotiable:

> language generation may decide **how to say** a semantic answer, not **what is true**.

If future fluency work requires a compact neural surface realizer, that must be evaluated as a separate hybrid product track rather than silently inserted into the pure non-neural APCN research architecture.
