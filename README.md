# APCN V0.19 — Online English + Arithmetic

Current release candidate: **0.19.0**.

APCN V0.19 deliberately pauses world-model expansion. The immediate goal is to make the language core useful and falsifiable before reconnecting it to a richer world model.

The V0.19 question is:

> Can APCN learn English arithmetic constructions online and generalize addition/subtraction to unseen numbers and newly taught words without backpropagation, gradient descent, an external LLM, or a pretrained language model?

```text
English arithmetic request
        ↓
learned construction evidence
        ↓
operation hypothesis
  ADD / SUB / reverse SUB
        ↓
explicit arithmetic execution
        ↓
natural answer
```

The arithmetic words themselves are not direct parser branches such as `if word == "plus": add`. A compact bootstrap curriculum is fed through the same online demonstration learner used for later teaching.

## What V0.19 can currently do

Examples from the finite CI gate:

```text
YOU: what is 37 plus 58?
APCN: 37 + 58 = 95.

YOU: calculate 80 minus 13
APCN: 80 - 13 = 67.

YOU: subtract 7 from 30
APCN: 30 - 7 = 23.

YOU: what is seven plus five
APCN: 7 + 5 = 12.
```

It can also learn a new arithmetic word from one demonstration:

```text
YOU: what is 11 dax 8?
APCN: I can see two numbers, but I have not learned enough English evidence to identify the requested operation.

YOU: remember that 2 dax 3 equals 5
APCN: Learned that construction from the example. Its current arithmetic meaning is add.

YOU: what is 11 dax 8?
APCN: 11 + 8 = 19.
```

Subtraction can be learned the same way:

```text
YOU: remember that 9 nerk 4 equals 5
APCN: Learned that construction from the example. Its current arithmetic meaning is sub.

YOU: please calculate 20 nerk 7
APCN: 20 - 7 = 13.
```

Unknown wording remains explicit:

```text
YOU: what is 9 florp 2?
APCN: I can see two numbers, but I have not learned enough English evidence to identify the requested operation.
```

## Desktop

Windows PowerShell:

```powershell
cd "D:\HUD Jarvis\APCN"
git checkout v0.19-language-math
git pull
.\.venv\Scripts\Activate.ps1
python run_desktop_v0_19.py
```

For a fresh environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_desktop_v0_19.py
```

The V0.19 desktop is intentionally focused. It shows whether memory came from a **LOADED CHECKPOINT** or a **CLEAN IN-MEMORY SESSION**, provides a **Start Clean Test Session** button that does not delete the saved checkpoint, and keeps the main interaction in one chat window.

## Finite V0.19 gate

Run:

```bash
python -m unittest tests.test_v0_19 -v
python benchmark_v0_19.py
```

The gate requires 100% on:

- unseen-number addition;
- unseen-number subtraction;
- English wrapper/operator recombination;
- reverse-order subtraction constructions;
- basic written number words;
- one-shot invented addition-word learning;
- one-shot invented subtraction-word learning;
- unknown-operator honesty;
- persistence of newly learned arithmetic language;
- V0.18 semantic fallback compatibility;
- no-backprop/no-external-model architecture checks.

See `V0_19_LANGUAGE_MATH.md` for the architecture and scientific boundary.

## What remains from V0.18

V0.19 inherits the V0.18 semantic conversation system as a fallback for non-arithmetic language. V0.18 provides:

- structured `PROPERTY` and `IS_A` truth queries;
- universal category inference;
- proof-aware `why?`;
- entity fact aggregation;
- causal, conditional, and temporal retrieval;
- explicit concept identity such as `fluxion -> acceleration`;
- bounded online prototype memory that is not truth authority;
- bounded answer-plan → English realization.

The older desktop remains available:

```bash
python run_desktop_v0_18.py
```

V0.18 example:

```text
YOU: remember that milo is a cat
APCN: Stored as explicit semantic memory: Milo is a cat.

YOU: remember that every cat is a creature
APCN: Stored as explicit semantic memory: Every cat is a creature.

YOU: is milo a creature?
APCN: Yes. Milo is a creature. I can derive that because Milo is a cat, and every cat is a creature.
```

## Scientific boundary

V0.19 is **not** a claim of LLM-level English competence. It is a finite first language milestone. Passing it proves that APCN can acquire and reuse a narrow class of English arithmetic constructions online without backpropagation.

The intended progression is now language-first:

```text
addition/subtraction
        ↓
multiplication/division
        ↓
comparison + variables
        ↓
multi-step arithmetic
        ↓
broader sentence semantics
        ↓
paraphrase/general English benchmarks
        ↓
reconnect the mature language core to the world model
```

This avoids spending indefinite effort on world-model plumbing while the core language capability remains brittle.

Historical architecture notes remain in `V0_15_COURSE_CORRECTION.md`, `V0_16_BIDIRECTIONAL_LANGUAGE.md`, `V0_17_STRUCTURED_DISCOURSE.md`, `V0_18_NATURAL_CONVERSATION.md`, and `V0_18_ONLINE_CONCEPT_LEARNING.md`.
