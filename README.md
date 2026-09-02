# APCN V0.9 — Grounded Semantic Language Studio

V0.9 keeps the V0.8.2 perceptual grounding system and adds the first contextual semantic-language layer: utterances are learned as mappings into language-independent semantic programs rather than as bags of visual words.

## Update and run

```bash
cd ~/APCN
git pull
source .venv/bin/activate
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python run_desktop_v0_9.py
```

The UI remains designed for **1366×768**.

## What V0.9 adds

- grounded binary relations
- statement / question / command intent (`ASSERT`, `QUERY`, `GOAL`)
- entity composition from learned properties/categories
- conjunction / grouping (`and`-like cues)
- sequence (`then`-like cues)
- negation
- contextual/pronoun reference to a salient entity
- arbitrary-vocabulary tests to verify that literal English strings are not required
- compact persistent semantic cue statistics
- separate perception and semantic-language testing

## Testing Ground UI change

The redundant score cards have been removed.

The Testing Ground now has:

```text
┌───────────────────────────────────────────────┐
│ confusion matrix       confusion matrix      │
├───────────────────────┬───────────────────────┤
│ FAILURES              │ GRAPHS                │
│ exact failed examples │ accuracy vs episodes  │
└───────────────────────┴───────────────────────┘
```

The graph stores benchmark history, so you can test at 400, 800, 1200, 1600, 2200… learned experiences and see whether learning is actually improving. Tests are read-only and show the learner episode count before and after.

## Recommended language curriculum

```text
0–399       lexical relations
400–799     statement / question / command intent
800–1199    composition + conjunction
1200–1599   sequence + negation
1600+       mixed language + contextual reference
```

Start with:

```text
Language + Context → Auto Train Language Batch → 800
Testing Ground → Language semantics → Run Test
```

Then continue to roughly 1600 and 2200+ and test again. Each result becomes another point on the graph.

## CLI language training

```bash
python train_language_v0_9.py --episodes 2600 --test-samples 500
```

Re-test a saved semantic memory:

```bash
python test_language_v0_9.py outputs/v0_9/semantic_memory_v0_9.json --samples 500
```

## Validation

```bash
python -m unittest discover -s tests -v
```

The V0.9 semantic regression suite passes 7/7 tests locally. A deterministic controlled synthetic benchmark reached 100% exact semantic-program accuracy by ~2200 curriculum steps on the training grammar. This is a controlled synthetic result, not open-domain English understanding. The held-out-template benchmark is materially harder and remains an explicit target for the next construction-learning work.

Full technical notes: `README_V0_9.md`.
