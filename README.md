# APCN V0.10 — Automatic Learning Studio

Current milestone: **grounded semantic language acquisition + concept-from-concept definitions**.

```bash
cd ~/APCN
git pull
source .venv/bin/activate
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python run_desktop_v0_10.py
```

The V0.10 UI is 1366×768-friendly and has four focused areas:

- **Perception** — live shape/color auto-training progress with pause/resume.
- **Language** — one automatic adaptive curriculum; no manual relation/group selector.
- **Definitions** — concept dependency graphs, grounding audits and executable derived definitions.
- **Testing Ground** — generated held-out semantic tests, confusion matrices, failures and learning graphs.

Headless training:

```bash
python train_v0_10.py --language-experiences 2800 --test-samples 600 --definitions
```

Tests:

```bash
python -m unittest discover -s tests -v
```

See `README_V0_10.md` for architecture, limitations and tutorial.
