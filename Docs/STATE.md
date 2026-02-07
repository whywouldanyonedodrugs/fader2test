# STATE (single source of truth)

Repo: `/opt/fader2`  
Working dir (machine): `/opt/fader2`

## Phase
Short-only strategy development is active in `shortonly/` with boom/stall/trigger signals and guarded sweeps.

## Non-negotiables
- No guessing: claims must be backed by code/docs/output.
- `upstream_longonly/` is frozen reference code and should not be edited.
- Local canary verification is required before merge (`tools/canary_all.sh` or equivalent `tools/check_canary.py` runs).
- CI mirrors local checks but does not replace local validation.

## Canary definition
Dataset:
- Source lake: `/opt/testerdonch/parquet` (local; not committed)
- Symbols: `symbols_canary.txt` (10 symbols)
- Window: `2024-01-01T00:00:00Z` .. `2024-06-01T00:00:00Z`

Runners:
- Primary: `tools/canary_all.sh`
- Underlying: `tools/check_canary.py`

## Recorded baseline (committed)
- `Docs/baselines/BASELINE_UPSTREAM_CANARY.json`
  - signals_rows=`457`, trades_rows=`59`

## Current runtime defaults (short sweep)
- Universe source: `shortonly/symbols.txt`
- Timeframe defaults: `shortonly/config.py` (`START_DATE="2023-01-01"`, `END_DATE="2025-11-15"`)
- Sweep entry point: `shortonly/sweep_mr_short_params_guarded.py`

## How to run the gate (local)
```bash
./tools/canary_all.sh
```
