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

## 1) Regenerate canary dataset (from lake)
rm -rf data_canary
python tools/make_canary_from_lake.py \
  --lake-dir /opt/testerdonch/parquet \
  --out-dir data_canary \
  --symbols $(tr '\n' ' ' < symbols_canary.txt) \
  --start "2024-01-01T00:00:00Z" \
  --end   "2024-06-01T00:00:00Z" \
  --ts-unit auto

## 2) Run upstream canary
rm -rf reports/upstream_signals/* reports/upstream_results/*
python tools/run_upstream_canary.py

## 3) Snapshot upstream (for diagnostics; baseline is committed separately)
python tools/snapshot_upstream_canary.py

## 4) Run shortonly canary
rm -rf reports/shortonly_signals/* reports/shortonly_results/*
python tools/run_shortonly_canary.py

## 5) Compare
python tools/compare_upstream_shortonly_canary.py
