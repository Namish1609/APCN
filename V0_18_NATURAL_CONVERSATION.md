# APCN V0.18 — Natural Semantic Conversation

V0.18 closes the gap between APCN's structured semantic reasoning and ordinary desktop conversation.

It was motivated by real V0.17 failures such as ordinary `is milo active?` bypassing structured property memory, `is milo a creature?` bypassing a universal rule, `what do you know about milo?` returning only a generic known-state response, standard `X causes Y` being stored in the wrong semantic shape, `why?` explaining the wrong memory path, and near-miss vocabulary producing generic parser errors.

## Architecture

```text
ordinary English
      ↓
NaturalConversationEngineV18
      ↓
semantic compiler
      ↓
SemanticClause
      ↓
StructuredSemanticMemoryV17 + ConceptStore
      ↓
transparent proof / lookup
      ↓
AnswerPlanV18
      ║
      ║ truth firewall
      ▼
NaturalRealizerV18
      ↓
English
```

The realizer receives an authorized `AnswerPlanV18`; it has no reference to semantic truth memory, the concept store, world state, or perception.

## Corrections

### Ordinary yes/no questions

`is milo active?`, `is milo a cat?`, and `is milo a creature?` now compile into the same explicit `PROPERTY` / `IS_A` structures used by V0.17 reasoning.

### Category/property separation

V0.17 mirrored both `IS_A` and `PROPERTY` teaching into the old V0.15 `is_a` triple store. V0.18 removes that unsafe bridge. `milo is active` remains a property; it is not converted into `milo IS_A active`.

### Standard causative construction

V0.18 structurally recognizes `<cause> causes <effect>`. Both sides are recursively parsed. The rule is lexical-independent rather than special-cased for the desktop sentence.

### Entity knowledge aggregation

`what do you know about milo?` now collects structured entity facts and may include rule-derived category facts.

### Proof-aware follow-up

After a derived answer, `why?` uses the proof that produced the immediately previous answer.

### Lexical repair

Unknown near-miss terms are not silently bound. APCN can ask `I don't know 'fluxation'. Did you mean 'fluxion'?` using bounded edit distance over already known terms. Confirmation is still required.

### Canonical concept binding

V0.18 checks that inherited definition, query, semantic responder, and conversation layers reference one canonical `ConceptStore`. Detached definition records from older checkpoint bindings are reconciled before language engines are rebuilt.

## Natural realization

V0.18 adds bounded semantic-plan → English constructions for explicit truth, derived truth, negation, unknown truth, entity summaries, causes, conditional consequences, temporal answers, proof explanations, definitions, lexical clarification, and semantic-teaching acknowledgements.

This is not an LLM and does not use token prediction. It is the first release on the planned generation path, not a claim of frontier-model fluency.

## Persistence

```text
outputs/v0_18/
  base_v17/
  natural_realization_v0_18.json
  session_v0_18.json
```

Raw conversation transcripts are not retained as long-term V0.18 memory.

## Desktop

Windows:

```powershell
cd "D:\HUD Jarvis\APCN"
git checkout main
git pull
.\.venv\Scripts\Activate.ps1
python run_desktop_v0_18.py
```

Use the inherited **Conversation** tab normally. V0.18 adds **Conversation Quality** with the desktop regression sequence and architecture audit.

## Evaluation boundary

The V0.18 transcript regression is a development regression derived from a real user session. It is explicitly **not** a blind general-English benchmark.

The release gate checks immediate alias lookup, ordinary property/category truth routing, entity-fact aggregation, universal inference, proof-aware `why?`, conditional/causal/temporal retrieval, lexical clarification, property/category separation, unknown honesty, natural realization diversity, truth isolation, and zero visual-training changes.
