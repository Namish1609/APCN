# APCN V0.17 — Structured Discourse Semantics

Current release candidate: **0.17.0**.

APCN is an experimental non-neural cognitive architecture built around persistent concepts, explicit world memory, semantic programs, bounded online learning, and inspectable reasoning. V0.17 continues the V0.15/V0.16 course correction: language is a **compiler and realizer around explicit cognition**, not a replacement for concept/world memory.

## V0.17 architecture

```text
English
   ↓
V0.16 bidirectional constructions
   ↓
V0.17 structured semantic compiler
   ↓
┌──────────────────────────────────────┐
│ explicit semantic structure          │
│                                      │
│ NOT(...)                             │
│ CAUSE(cause, effect)                 │
│ IF(condition, consequence)           │
│ BEFORE(first, second)                │
│ AFTER(first, second)                 │
│ AND(left, right)                     │
│ FORALL_ISA(kind, category)           │
│ EXISTS_PROPERTY(kind, property)      │
│ atomic entity/property/event facts   │
└──────────────────────────────────────┘
   ↓
concept/world memory + explicit semantic memory
   ↓
transparent reasoning
   ↓
semantic answer
   ↓
the same operator construction memory
   ↓
English
```

The truth firewall remains mandatory. The V0.17 surface generator receives an explicit `SemanticClause` and has **no reference to semantic truth memory, the world model, perception memory, or the concept store**. It can choose how to express a proposition but cannot manufacture factual content.

V0.17 remains non-neural: no transformer, external LLM, gradient descent, backpropagation, or trainable neural language model is used.

## Why V0.17 exists

V0.16 proved a controlled `English ↔ SemanticFrame` path for flat requests and answers. V0.17 expands the **meaning representation itself** instead of increasing sentence-template count.

The release adds:

- recursive `SemanticClause` structures;
- explicit negation;
- explicit causal relations;
- explicit conditionals;
- temporal `BEFORE` / `AFTER` structure;
- conjunction;
- universal category rules;
- existential property statements;
- tense on atomic events/properties;
- bounded semantic discourse focus for references such as `it`;
- bounded structured semantic memory;
- transparent universal-rule, conditional, causal and temporal lookup;
- bidirectional learned operator constructions;
- semantic roundtrip testing for nested structures;
- V0.16 checkpoint migration;
- a **Structured Semantics** desktop lab;
- a dedicated V0.17 CI/research gate.

## Example conversation

```text
YOU:  remember that milo is a cat
APCN: Stored as explicit semantic memory: Milo is a cat.

YOU:  remember that it is active
APCN: Stored as explicit semantic memory: Milo is active.

YOU:  is it true that it is active?
APCN: Yes. Milo is active.
```

The raw transcript is not retained as long-term memory. V0.17 keeps bounded semantic discourse state such as the current focus entity and recent semantic proposition.

A universal-rule example:

```text
YOU:  remember that every cat is a creature
YOU:  remember that milo is a cat
YOU:  is it true that milo is a creature?
APCN: Yes. Milo is a creature.
```

The answer is derived from explicit records rather than from language co-occurrence.

A conditional example:

```text
YOU:  remember that if battery is empty then device stops
YOU:  what follows if battery is empty?
APCN: ... Device stops.
```

A causal example:

```text
YOU:  remember that lamp turns off because power fails
YOU:  what causes lamp turns off?
APCN: ... Power fails.
```

A temporal example:

```text
YOU:  remember that door opens before light turns on
YOU:  what happens before light turns on?
APCN: ... Door opens.
```

Negation remains explicit rather than closed-world guesswork:

```text
YOU:  remember that not sensor is active
YOU:  is it true that sensor is active?
APCN: No. My explicit memory supports the negation ...
```

If neither a proposition nor its negation is known, APCN reports **unknown** instead of assuming false.

## Language generation contract

Higher-order operators are learned as reusable bidirectional constructions. For example, the same semantic structure:

```text
CAUSE(
  EVENT(power, fail),
  EVENT(lamp, turn_off)
)
```

can be parsed from or realized through several learned causal surfaces. Nested structures are generated recursively and then parsed back for semantic roundtrip evaluation.

This release does **not** claim that these controlled constructions equal open-domain English fluency. The objective is semantic compositionality and inspectable truth separation.

## Persistence

V0.17 saves to `outputs/v0_17/`:

```text
base_v16/
operator_constructions_v0_17.json
semantic_memory_v0_17.json
semantic_discourse_v0_17.json
session_v0_17.json
```

Raw chat sentences are not stored in the V0.17 semantic/discourse memories. Structured records, evidence counts, and bounded discourse entities are persisted.

## Desktop usage

Windows PowerShell:

```powershell
cd "D:\HUD Jarvis\APCN"
git checkout main
git pull
.\.venv\Scripts\Activate.ps1
python run_desktop_v0_17.py
```

Linux/X11:

```bash
cd ~/APCN
git checkout main
git pull
source .venv/bin/activate
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python run_desktop_v0_17.py
```

The inherited **Conversation** and **Bidirectional Language** tabs remain available. V0.17 adds **Structured Semantics**, where you can:

- parse a sentence into nested semantic structure;
- inspect parser evidence;
- store an explicit semantic proposition/rule;
- generate language from explicit nested semantics;
- run semantic roundtrip checks;
- inspect semantic memory and discourse state.

## Testing

Fast V0.17 gate:

```bash
python -m unittest tests.test_v0_17 -v
python benchmark_v0_17.py
```

Full project regression:

```bash
python -m unittest discover -s tests -v
```

The V0.17 controlled architecture gate measures:

- structured operator parsing;
- nested semantic roundtrip preservation;
- generation diversity for higher-order operators;
- discourse reference resolution;
- universal category inference;
- conditional consequence lookup;
- causal retrieval;
- temporal-order retrieval;
- explicit negation;
- unknown honesty;
- truth/generation separation;
- zero visual-training changes.

These are controlled architecture/composition tests, **not** a blind proof of general English competence or LLM-level fluency.

## Scientific boundaries

V0.17 does **not** claim:

- unrestricted English grammar;
- LLM-level sentence generation;
- broad commonsense reasoning;
- automatic acquisition of unknown word meanings from nothing;
- general programming ability;
- general intelligence;
- superiority to modern neural language models.

The research question is narrower:

> Can APCN progressively expand an explicit, grounded cognitive system from flat language frames into nested discourse semantics and transparent rule reasoning, while keeping factual knowledge outside the language generator?

That separation remains non-negotiable. If future product-quality fluency requires a compact neural surface realizer, it must be evaluated as a separate hybrid product track rather than silently changing the claims of the pure APCN research architecture.

See `V0_15_COURSE_CORRECTION.md`, `V0_16_BIDIRECTIONAL_LANGUAGE.md`, and `V0_17_STRUCTURED_DISCOURSE.md` for the successive language architecture contracts.
