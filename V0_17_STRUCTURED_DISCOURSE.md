# APCN V0.17 — Structured Discourse Semantics

V0.17 deliberately does not chase broader English by adding more surface phrases. It extends the explicit semantic representation underneath language.

## Architectural contract

```text
surface language
      ↓
learned operator constructions + atomic grammar
      ↓
SemanticClause
      ↓
explicit semantic / concept / world memory
      ↓
transparent reasoning
      ↓
SemanticClause answer
      ↓
learned operator constructions
      ↓
surface language
```

The language realizer has no direct truth-memory reference. World facts and structured semantic records are owned upstream.

## Semantic operators

V0.17 introduces recursive clauses for:

- `NOT(P)`
- `CAUSE(C, E)`
- `IF(C, E)`
- `BEFORE(A, B)`
- `AFTER(A, B)`
- `AND(A, B)`
- `FORALL_ISA(kind, category)`
- `EXISTS_PROPERTY(kind, property)`
- atomic `IS_A`, `PROPERTY`, `RELATION`, and `EVENT` propositions

Atomic clauses also carry a small explicit tense field (`present`, `past`, `future`, or `atemporal`).

## Memory

`StructuredSemanticMemoryV17` is bounded and stores semantic clauses rather than raw teaching sentences. Repeated identical propositions collapse into one record with support and provenance.

`SemanticDiscourseV17` is a bounded working discourse registry. It tracks semantic focus and entity salience so a reference such as `it` can resolve to the current entity without retaining a raw permanent chat transcript.

## Inference boundary

V0.17 intentionally supports only a small transparent reasoning set:

- exact proposition lookup;
- explicit negation lookup;
- one-hop universal category inheritance;
- conditional consequence lookup;
- causal explanation lookup;
- temporal-before lookup.

This is not a general theorem prover. New inference procedures should be added only when they have explicit semantics, tests, provenance, and bounded behavior.

## Evaluation policy

The V0.17 benchmark is a controlled architecture/composition gate. It is not described as a blind general-English benchmark.

The gate checks semantic structure, nested parse/generate roundtrip, discourse reference, rule reasoning, unknown honesty, memory isolation and zero visual-training changes. Failures should be fixed at the grammar/semantic mechanism level rather than by adding the exact failed sentence as a special rule.

## Product boundary

The pure APCN research track remains non-neural. A future product may separately evaluate a compact neural surface realizer for fluency, but that module must not become the factual or reasoning substrate and must not be presented as evidence that the pure non-neural language generator itself achieved LLM-level fluency.
