#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-headless.txt
printf '\nAPCN V0.8 headless installed.\nRun: source .venv/bin/activate && python train_concepts_v0_8.py\n'
