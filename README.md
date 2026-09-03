# APCN V0.10.1 — Automatic grounded learning + explicit concept query

Current release: **0.10.1**.

V0.10.1 keeps V0.10's automatic grounded-language curriculum and concept-from-concept definitions, then adds the missing evaluation and interaction layer:

- live progress for perception and language training;
- automatic semantic-language curriculum;
- color + shape read-only testing with confusion matrices, failures and graphs;
- generated held-out language testing;
- APCN firing/spiking visualization in Perception, Language and Definitions;
- **Ask APCN** concept-memory interface (`what is acceleration?`);
- executable concept queries (`calculate acceleration if velocity change = 20 and time = 4`);
- improved learned intent constructions up to five-token prefixes.

## Run the desktop studio

```bash
cd ~/APCN
git pull
source .venv/bin/activate
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python run_desktop_v0_10_1.py
```

The compatibility launcher also opens the latest patch:

```bash
python run_desktop_v0_10.py
```

## Tests

```bash
python -m unittest discover -s tests -v
```

See `README_V0_10_1.md` for the V0.10.1 workflow and scientific boundaries, and `README_V0_10.md` for the V0.10 architecture.
