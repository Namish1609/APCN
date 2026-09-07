# APCN V0.16 — Bidirectional Semantic Language

Current release candidate: **0.16.0**.

APCN is an experimental non-neural cognitive architecture built around persistent concepts, explicit world memory, semantic programs, bounded online learning, and inspectable reasoning. V0.16 follows the V0.15 course correction: language is treated as a **compiler and surface realizer around the concept/world system**, not as the primary knowledge substrate.

## V0.16 architecture

```text
English input
    ↓
learned bidirectional constructions
    ↓
semantic request frame
    ↓
concept / world memory
    ↓
explicit reasoning
    ↓
semantic answer frame
    ↓
the same construction memory
    ↓
English surface realization
```

The semantic responder is the truth firewall. The surface generator receives a semantic frame and has **no direct reference to the concept store, fact memory, perception memory, or world model**. It decides how to express an answer, not what is true.

V0.16 remains non-neural: no transformer, external LLM, gradient descent, backpropagation, or trainable neural language model is used.

## What changed from V0.15

V0.15 established conversational semantic routing over explicit memory and removed the experimental raw-corpus/n-gram language-exposure path. V0.16 adds:

- an explicit `SemanticFrame` intermediate representation;
- a bounded `BidirectionalConstructionMemory`;
- shared constructions for both parsing and generation;
- multiple surface realizations from one semantic frame;
- `parse(generate(S))` semantic roundtrip testing;
- a semantic responder that alone can access concept/fact truth;
- a compositional request parser that recombines lexical functions learned from TRAIN constructions;
- conservative generic morphology normalization for ordinary inflections such as `depend/depends`;
- V0.15 explicit teaching and memory migration;
- a **Bidirectional Language** desktop lab;
- a dedicated V0.16 CI/research gate.

## Why the language generator is separated from knowledge

For a semantic answer such as:

```text
STATE_DEFINITION(
    concept = acceleration,
    definition = "velocity change divided by time"
)
```

V0.16 can realize several stored constructions, for example:

```text
Acceleration is velocity change divided by time.
Put simply acceleration is velocity change divided by time.
The stored definition of acceleration is velocity change divided by time.
I understand acceleration as velocity change divided by time.
```

The realizer cannot independently add a claim about force, gravity, density, or any other concept unless that information is present in the semantic frame passed to it.

This is the central V0.16 architectural contract:

> **Generation may choose how to say a semantic answer; it may not decide what is true.**

## Compositional language test

The first exact construction implementation passed semantic generation/roundtrip tests but scored 0% on an original DEV set containing genuinely unseen lexical material. That result is retained as a limitation rather than patched phrase-by-phrase.

A separate recombination split was frozen before first execution. Its words/semantic cues occur in TRAIN, but their syntax is recombined into new forms. A generic morphology layer normalizes productive inflections without assigning dialogue semantics. This tests a narrower, scientifically defensible question:

> Can APCN recombine linguistic functions it has already learned into unseen syntax?

The release benchmark reports this separately from the original DEV score. Neither is presented as proof of general English competence or LLM-level fluency.

## Persistent teaching

V0.16 keeps V0.15's explicit semantic-memory teaching path. For example:

```text
fluxion means acceleration
```

updates lexical-semantic memory immediately. A later request such as:

```text
what is fluxion
```

can resolve through the alias into the stored `ACCELERATION` concept without global retraining.

Likewise, explicit facts and supported concept-from-concept definitions remain persistent knowledge rather than dialogue statistics.

## Desktop usage

Windows PowerShell:

```powershell
cd "D:\HUD Jarvis\APCN"
git checkout main
git pull
.\.venv\Scripts\Activate.ps1
python run_desktop_v0_16.py
```

Linux/X11:

```bash
cd ~/APCN
git checkout main
git pull
source .venv/bin/activate
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python run_desktop_v0_16.py
```

The inherited **Conversation** tab uses the V0.16 semantic path when supported and falls back to the V0.15 conversational compiler for operations V0.16 does not yet represent.

The new **Bidirectional Language** tab lets you:

- parse a sentence into a semantic frame;
- inspect parser evidence;
- generate multiple paraphrases from the same frame;
- run semantic roundtrip checks;
- inspect the bidirectional construction memory and architecture audit.

## Testing

Fast V0.16 gate:

```bash
python -m unittest tests.test_v0_16 -v
python benchmark_v0_16.py
```

Full project regression:

```bash
python -m unittest discover -s tests -v
```

The V0.16 benchmark gates:

- supported semantic request/response accuracy;
- at least five surface variants for supported answer-frame families;
- semantic roundtrip preservation;
- explicit unknown handling;
- transfer of explicit teaching into V0.16 reasoning;
- generation content isolation;
- recombination of learned linguistic functions;
- zero visual-training changes.

The benchmark explicitly labels itself **development/architecture-contract**, not a blind final English benchmark.

## Scientific boundaries

V0.16 does **not** claim:

- LLM-level natural-language generation;
- broad open-domain English understanding;
- internet-scale language learning;
- autonomous programming ability;
- general intelligence;
- superiority to modern neural language models.

It is testing a narrower architectural hypothesis:

> Can a bounded non-neural language system map multiple surface constructions into explicit semantic operations and realize explicit semantic answers back into varied language, while factual knowledge remains in a separate persistent concept/world memory?

That separation is mandatory for future APCN releases. If future product-quality fluency ultimately requires a compact neural surface realizer, it must be evaluated as a separate hybrid product track rather than silently changing the claims of the pure APCN research architecture.

See `V0_15_COURSE_CORRECTION.md` for the language course correction and `V0_16_BIDIRECTIONAL_LANGUAGE.md` for the V0.16 design contract. Historical V0.14/V0.13/V0.12 documentation remains in the repository.
