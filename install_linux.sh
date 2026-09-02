#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo
echo "APCN V0.7 environment ready."
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Train with:    python train_concepts_v0_7.py --episodes 2400"
