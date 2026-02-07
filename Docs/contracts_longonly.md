# Extracted contracts from long-only manual (ground truth)

This file is a condensed checklist of non-negotiable contracts extracted from Docs/bull_longonly_manual.md.
Only add items that are confirmed by upstream code OR are explicitly stated in the manual.

## Data inputs
- 5m parquet path pattern: `<PARQUET_DIR>/<SYMBOL>.parquet`
- required columns: `open`, `high`, `low`, `close`, `volume`
- optional columns used by short stack: `funding_rate`, `open_interest`
- timestamp requirements: file must expose a time-like field (`timestamp`, `time`, etc.) or DatetimeIndex
- normalization: timestamps are converted to UTC and sorted/deduped

## Signal dataset contract
- output layout: parquet dataset partitioned by `symbol` under `signals/symbol=*`
- required base columns: `timestamp`, `symbol`, `entry`, `atr`, `side`
- additional fields commonly present: `don_break_level`, `pullback_type`, `entry_rule`, `rs_pct`, `funding_z`
- merge keys for meta/offline joins: (`timestamp`, `symbol`) with controlled rounding/tolerance

## Trade record schema
- canonical fields: `entry_ts`, `exit_ts`, `entry`, `exit`, `qty`, `side`, `sl`, `tp`, `pnl`, `pnl_R`, `fees`
- diagnostics: `mae_over_atr`, `mfe_over_atr`, lock/cooldown timeline in separate `lock_timeline.csv`
- regime fields: include regime state/probability columns when available
- meta fields: include `meta_p` and derived gating/sizing diagnostics when enabled

## Execution semantics (long-only baseline)
- decision timestamp: signal timestamps are UTC-normalized and processed in time order
- fill semantics: deterministic bar-based simulation with configurable spread
- touch rules and tie-break: deterministic policy with optional 1m intrabar resolver

## Meta-model contract (baseline)
Offline merge:
- rounding/tolerance/dedup follow `META_MERGE_ROUND`, `META_MERGE_TOL`, and (`ts`,`sym`) aggregation

Online scoring:
- strict schema behavior is controlled by `META_STRICT_SCHEMA`
- required artifacts include model + manifest + calibration config
- calibration supports none/sigmoid/isotonic

## Known hazards / required fixes
- avoid future leakage in any regime/meta feature construction
- keep dependency/runtime parity between local dev and server execution environments
