# Commands (copy/paste)
## “Start here” reading list for any new chat/agent
- Docs/STATE.md
- Docs/baselines/CANARY_WORKFLOW.md
- .ai/PARITY.md
- .ai/COMMANDS.md
- Docs/short_only_spec.md


## 0) Environment
cd /opt/fader2
source .venv/bin/activate

## 0.5) One-command canary (recommended)
./tools/canary_all.sh
# Rebuild data_canary from lake:
./tools/canary_all.sh --regen
# Skip baseline check (debug only):
./tools/canary_all.sh --skip-baseline

## 1) Regenerate canary dataset (from lake)
rm -rf data_canary
./.venv/bin/python tools/make_canary_from_lake.py \
  --lake-dir /opt/testerdonch/parquet \
  --out-dir data_canary \
  --symbols $(tr '\n' ' ' < symbols_canary.txt) \
  --start "2024-01-01T00:00:00Z" \
  --end   "2024-06-01T00:00:00Z" \
  --ts-unit auto

## 2) Run upstream canary
rm -rf reports/upstream_signals/* reports/upstream_results/*
./.venv/bin/python tools/run_upstream_canary.py

## 3) Snapshot upstream (for diagnostics; baseline is committed separately)
./.venv/bin/python tools/snapshot_upstream_canary.py

## 4) Run shortonly canary
rm -rf reports/shortonly_signals/* reports/shortonly_results/*
./.venv/bin/python tools/run_shortonly_canary.py

## 5) Compare
./.venv/bin/python tools/compare_upstream_shortonly_canary.py

## 6) Gate (REQUIRED before PR/merge)
# Phase-0 parity (current):
./.venv/bin/python tools/check_canary.py --mode parity --clean
# If/when Phase-1 baseline mode is enabled for shortonly:
# ./.venv/bin/python tools/check_canary.py --mode baseline --target shortonly --baseline Docs/baselines/BASELINE_SHORTONLY_CANARY.json

## 7) Full short sweep (long run; use tmux)
rm -rf shortonly/parquet
ln -s /opt/fader2/parquet shortonly/parquet
ls -1 shortonly/parquet | grep '\.parquet$' | sed 's/\.parquet$//' > shortonly/symbols.txt
wc -l shortonly/symbols.txt
for arc in breakdown_retest_vwap avwap_lower_high_reject funding_spike_crowding_fade; do
  echo ">>> STARTING SWEEP: $arc"
  ./.venv/bin/python shortonly/sweep_mr_short_params_guarded.py --archetype "$arc"
done
