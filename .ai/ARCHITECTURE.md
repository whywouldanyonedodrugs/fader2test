# FADER2 architecture (long-only upstream -> short-only refactor)

## Purpose
Refactor/port a proven long-only system into a clean short-only backtester while preserving tested mechanics (execution, throttles, meta gating/sizing, data contracts) and preventing silent drift.

## Code zones
1) `upstream_longonly/` (read-only snapshot)
- Imported from the bull/long-only repo.
- Never modified.
- Used only for characterization parity and contract reference.

2) `fader2/` (refactored implementation)
- New canonical code.
- Eventually short-only.
- May include a temporary long-compat mode to validate refactor correctness.

## Critical sub-systems (must be preserved mechanically)
A) Data loader (5m; optional 1m)
- Timestamp normalization to UTC, consistent index/column handling, date slicing, drop-last-partial semantics.

B) Signal generation (will become short-only)
- Long-only donchian breakout/retest becomes lower-breakout/retest from below.

C) Execution simulator (must be side-correct)
- For shorts: SL checks highs; TP checks lows; bid/ask/slippage conventions must be explicit.
- Intrabar tie-breaking (1m) must be supported if upstream supports it.

D) Meta-model gating and sizing (high importance)
- Two paths must remain well-defined:
  - Offline merge-asof of `meta_p`
  - Online scoring via exported bundle with strict feature manifest enforcement
- Do not reuse long-signed feature names with inverted semantics under the same name.

E) Regimes
- Any regime feature used for training/gating must be past-only (no smoothed probabilities / lookahead).
- 4h filtered Markov is acceptable; daily smoothed Markov is not.

## Output artifacts (target)
Each run writes to `reports/<run_id>/`:
- `config_snapshot.json`
- `metrics.json`
- `trades.parquet` (canonical schema)
- optional curves: `equity_curve.parquet`, `positions.parquet`

## Refactor stages (safe sequence)
Stage 0: Import upstream long-only code into `upstream_longonly/` + restore missing dependencies.
Stage 1: Build Tier A invariants + Tier B canary run around upstream to lock baseline.
Stage 2: Implement `fader2/` with long-compat mode and prove parity to upstream on canary.
Stage 3: Introduce short-only signal + short-only simulator while keeping meta-model infrastructure intact.
Stage 4: Train/export bear meta bundle with a new manifest; update online scoring and regression gates.
Stage 5: Remove long-compat mode if desired; short-only becomes the single canonical path.
