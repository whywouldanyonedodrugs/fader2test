# Canary workflow (Phase-1 Completed)

## Unified Gate
The primary entry point for all safety checks is now:

bash tools/canary_all.sh

This script automatically:
1. Runs the Parity Check (Upstream vs Shortonly).
2. Runs the Baseline Check (Upstream vs Committed JSON).
3. Fails fast if any step errors.

## CI / Automation
- A self-hosted GitHub Actions runner is configured to run `tools/canary_all.sh` on every push to `main`.
- The runner executes on the data machine, ensuring access to the local `data_canary` lake.

## Underlying Tools (Advanced Usage)
If `canary_all.sh` fails, you can debug using the individual tools:

### Run the Phase-0 parity gate (fast regression)
python tools/check_canary.py --mode parity

### Validate against committed baseline
python tools/check_canary.py --mode baseline --target upstream --baseline Docs/baselines/BASELINE_UPSTREAM_CANARY.json

### Regenerate canary data slice (server-only)
python tools/make_canary_from_lake.py \
  --lake-dir /opt/testerdonch/parquet \
  --out-dir data_canary \
  --symbols $(tr '\n' ' ' < symbols_canary.txt) \
  --start "2024-01-01T00:00:00Z" \
  --end   "2024-06-01T00:00:00Z" \
  --ts-unit auto

## Process for Phase 2 (Refactoring)
When we begin changing `shortonly` logic:
1. Run `tools/canary_all.sh` (Parity will fail as intended).
2. Verify `upstream` baseline still passes (regression check).
3. Create a NEW baseline for the new behavior:
   python tools/check_canary.py --mode baseline --target shortonly --write-baseline Docs/baselines/BASELINE_SHORTONLY_V2.json
