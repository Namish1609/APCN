# APCN V0.10.1 — Query + dual testing + firing graphs

V0.10.1 is a focused patch on V0.10.

## Changes

- **Ask APCN** tab for explicit concept-memory questions such as `what is acceleration?`.
- Executable queries such as `calculate acceleration if velocity change = 20 and time = 4`.
- Unknown concepts are reported as unknown; the query interface does not silently use an external LLM.
- **Colors + Shapes** read-only Testing Ground with color/shape confusion matrices, failures and a learning curve.
- Existing generated language testing remains read-only.
- Animated APCN firing/spiking visualization in **Perception**, **Language**, and **Definitions** training panels.
- Intent construction fix: learned sentence-prefix constructions are scored up to five tokens, improving held-out forms such as `it is the case that ...` and `can you determine whether ...` without hardcoding those English phrases to intent labels.

The firing graph visualizes sparse activation of concept/cue/feature nodes. It is not a claim that APCN uses biological neurons or a conventional neural network.

## Run

```bash
cd ~/APCN
git pull
source .venv/bin/activate
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python run_desktop_v0_10_1.py
```

`python run_desktop_v0_10.py` is also updated to launch the latest V0.10.1 UI.

## Testing suggested workflow

1. Perception: train 500–2000 experiences.
2. Testing Ground → Colors + Shapes → run 500 samples.
3. Language: train 2400–4800 experiences.
4. Testing Ground → Language → run 600 generated held-out samples.
5. Definitions: click **Learn science definition curriculum**.
6. Ask APCN: `what is acceleration?`, `what does force depend on?`, or `calculate acceleration if velocity change = 20 and time = 4`.

## Scientific boundary

The Ask interface is not yet a general conversational LLM. It is deliberately narrower: natural-language patterns are mapped to inspect/define/dependency/evaluate operations over APCN's explicit ConceptStore. This makes the test falsifiable and prevents hidden general-language knowledge from masking missing APCN concepts.
