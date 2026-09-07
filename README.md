# APCN V0.18 — Natural Semantic Conversation

Current release: **0.18.0**.

APCN is an experimental non-neural cognitive architecture built around persistent concepts, explicit world/semantic memory, bounded online learning, inspectable reasoning, and a language layer that compiles to and realizes explicit meaning.

V0.18 fixes the conversation-integration failures exposed by real V0.17 desktop use and begins the planned natural-generation phase.

```text
English
  ↓
semantic compiler
  ↓
SemanticClause
  ↓
concept/world + structured semantic memory
  ↓
transparent reasoning / proof
  ↓
AnswerPlanV18
  ║
  ║ truth firewall
  ▼
bounded natural realizer
  ↓
English
```

The surface realizer does **not** receive the truth memory, concept store, world state, or perception memory. It can choose how to say an authorized answer; it cannot decide what is true.

## What V0.18 fixes

- ordinary `is Milo active?` now queries structured `PROPERTY` memory;
- ordinary `is Milo a cat?` now queries structured `IS_A` memory;
- `is Milo a creature?` can use universal category inference;
- `why?` follows the proof for the immediately previous answer;
- `what do you know about Milo?` aggregates structured entity facts;
- standard `<cause> causes <effect>` is parsed compositionally;
- `PROPERTY` is no longer mirrored into the legacy `is_a` fact store;
- unknown near-miss vocabulary can trigger a suggestion without auto-learning;
- inherited concept/query/language bindings are reconciled to one canonical `ConceptStore`;
- V0.18 adds bounded semantic answer-plan → English realization constructions.

Example:

```text
YOU: remember that milo is a cat
APCN: Stored as explicit semantic memory: Milo is a cat.

YOU: remember that milo is active
APCN: Stored as explicit semantic memory: Milo is active.

YOU: is milo active?
APCN: Yes. Milo is active.

YOU: remember that every cat is a creature
APCN: Stored as explicit semantic memory: Every cat is a creature.

YOU: is milo a creature?
APCN: Yes. Milo is a creature. I can derive that because Milo is a cat, and every cat is a creature.

YOU: why?
APCN: Because Milo is a cat, and every cat is a creature.
```

Unknown remains explicit:

```text
YOU: is zorbin a creature?
APCN: I don't know whether Zorbin is a creature yet. I have no explicit fact or rule that establishes it.
```

## Desktop

Windows PowerShell:

```powershell
cd "D:\HUD Jarvis\APCN"
git checkout main
git pull
.\.venv\Scripts\Activate.ps1
python run_desktop_v0_18.py
```

Linux/X11:

```bash
cd ~/APCN
git checkout main
git pull
source .venv/bin/activate
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python run_desktop_v0_18.py
```

Use **Conversation** as the main interface. **Conversation Quality** shows the V0.18 architecture audit and the real desktop regression sequence.

## Testing

```bash
python -m unittest tests.test_v0_18 -v
python benchmark_v0_18.py
python -m unittest discover -s tests -v
```

The V0.18 benchmark is a **desktop transcript regression and architecture gate, not a blind general-English benchmark**.

## Scientific boundary

V0.18 does not claim LLM-level fluency, broad commonsense, programming ability, or superiority to modern neural language models. It tests a narrower question:

> Can ordinary conversational English be routed into explicit semantic memory and transparent reasoning, then expressed naturally enough through a bounded truth-isolated realization system?

No external LLM, transformer, gradient descent, backpropagation, or neural language model is used in the V0.18 language path.

See `V0_15_COURSE_CORRECTION.md`, `V0_16_BIDIRECTIONAL_LANGUAGE.md`, `V0_17_STRUCTURED_DISCOURSE.md`, and `V0_18_NATURAL_CONVERSATION.md`.
