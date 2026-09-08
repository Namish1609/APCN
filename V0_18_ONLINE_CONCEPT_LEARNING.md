# APCN V0.18 — Online Concept Identity and Prototype Learning

This extension to V0.18 addresses the failure exposed by teaching a new lexical symbol such as `fluxion means acceleration` and then discovering that the system could retrieve the alias but did not consistently reuse the underlying concept in new semantic propositions.

## Architectural contract

```text
ordinary language
      ↓
exact learned concept identity canonicalization
      ↓
V0.18 natural semantic compiler
      ↓
SemanticClause
      ↓
StructuredSemanticMemoryV17 / ConceptStore   ← truth authority
      ↓
transparent reasoning
      ↓
AnswerPlanV18 → NaturalRealizerV18

explicit semantic observations + ConceptStore definitions
      ↓
OnlinePrototypeMemoryV18                     ← similarity only, never truth
```

V0.18 therefore separates three things that were previously too easy to conflate:

1. **Lexical memory** — how a user refers to a concept.
2. **Concept identity** — which canonical concept the symbol denotes.
3. **Prototype state** — a bounded distributed representation accumulated from APCN's own explicit semantic experience.

## One-shot concept identity

Teaching:

```text
fluxion means acceleration
```

creates the language-facing lexical link and mirrors it into an explicit V0.18 concept identity binding:

```text
fluxion → acceleration
```

For semantic reasoning, an exact learned alias is canonicalized before parsing/reasoning. A proposition such as:

```text
is fluxion active?
```

can therefore be evaluated as a proposition about the canonical `acceleration` concept. APCN does **not** copy all acceleration facts into a second `fluxion` fact bucket.

Near-miss vocabulary remains conservative. `Fluxation` may trigger `Did you mean fluxion?`, but it is not silently stored as a new alias.

## Online prototype memory

`OnlinePrototypeMemoryV18` maintains a 256-dimensional bounded prototype for known concepts. Updates are generated online from:

- explicit `SemanticClause` roles and operators;
- properties, predicates, events and peer concepts;
- explicit ConceptStore dependency structure;
- explicit concept-identity teaching.

The representation uses deterministic signed feature hashing. It does not use a pretrained embedding model, transformer, external LLM, raw chat transcript, or raw text corpus.

The prototype layer currently supports similarity and nearest-concept inspection. It has no `infer_truth` path and no reference to semantic/world truth memory. Similarity cannot create, confirm, negate or overwrite a fact.

## Persistence

V0.18 checkpoints now include:

```text
outputs/v0_18/
  base_v17/
  natural_realization_v0_18.json
  concept_bindings_v0_18.json
  online_prototypes_v0_18.json
  session_v0_18.json
```

Raw conversation text is not retained in V0.18 long-term session history.

## Regression gate

The V0.18 gate now additionally requires:

- immediate `fluxion → acceleration` concept identity;
- structured truth transfer through the alias without fact copying;
- alias/canonical identity in prototype space;
- prototype truth-firewall isolation;
- checkpoint persistence of concept identity and prototypes;
- near-miss honesty (`Fluxation` is suggested, not silently bound);
- zero new visual-training changes.

This is a controlled architecture gate, not a claim of LLM-level open-English competence. It proves a narrower but important capability: **teach a new symbol once, bind it to an existing semantic concept, reuse that concept in new reasoning, and update a persistent distributed representation without making retrieval or vector similarity the truth substrate.**

## Next research milestone

The next version should move from exact concept identity toward **hypothesis-based generalization**: multiple candidate interpretations scored by symbolic compatibility, prototype similarity, discourse context and uncertainty, followed by explicit consistency checking. That is the bridge toward typo recovery, paraphrase transfer, ambiguous intent correction, code semantics and multimodal state inference without devolving into phrase-by-phrase rules.
