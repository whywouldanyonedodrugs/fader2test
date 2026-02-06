#!/bin/bash
set -euo pipefail

# Ensure we are in repo root
cd "$(dirname "$0")/.."

echo ">>> [1/2] Running Parity Check (Upstream vs Shortonly)..."
# Passes arguments (like --regen) to the parity check
python tools/check_canary.py --clean --mode parity "$@"

echo ""
echo ">>> [2/2] Running Baseline Check (Upstream vs Committed JSON)..."
# Strictly checks against the committed baseline file
python tools/check_canary.py --mode baseline --target upstream --baseline Docs/baselines/BASELINE_UPSTREAM_CANARY.json

echo ""
echo "SUCCESS: All canary gates passed."
