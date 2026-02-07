# Short-only behavioral spec (must be explicit)

This spec defines the active short-only behavior in `shortonly/`.

## Signal (short)
- Pattern: boom -> stall -> trigger.
- Boom: return over `BOOM_WIN` must exceed `BOOM_THRESH`.
- Stall: prior `STALL_WIN` bars must remain compressed (`range/ATR <= STALL_ATR_MAX`).
- Trigger:
  - `stall_break`: close breaks below stall low, or
  - `ema_cross`: fast EMA crosses below slow EMA.
- Optional funding gate: require funding z-score above configured threshold when enabled.
- RS/liquidity:
  - weekly RS and liquidity filter are applied when enabled in config.
  - symbols without valid parquet schema are dropped.

## Features (directional semantics)
Rule: directional features should either be side-specific names or side-invariant with explicit sign conventions.

Current key directional features:
- `don_break_level`: stored as stall low reference used for short trigger context.
- `don_dist_atr`: `(entry - don_break_level) / atr` (short context still uses positive distance above break level).
- `funding_z`: higher positive values represent more crowded long carry.

## Execution (short)
- Entry basis: signal bar close (`entry` from scout row).
- Stops/targets:
  - short SL is above entry (`entry + SL_ATR_MULT * ATR`)
  - short TP is below entry (`entry - TP_ATR_MULT * ATR`)
- Touch checks in simulator:
  - SL: ask high >= SL
  - TP: ask low <= TP
- Supports optional partial TP and trailing stop after TP1.
- Optional intrabar 1m tie-break via `USE_INTRABAR_1M`; default is deterministic `TIE_BREAKER`.

## Risk & throttles
- Position sizing supports percent or fixed cash risk modes.
- Throttles include cooldown/dedup/day caps/max open positions.
- Guardrails include:
  - max trades per variant
  - minimum equity fraction abort
  - minimum ATR% of price filter

## Outputs
- Trades: `results/trades.csv` and `results/trades.parquet` with `side="short"` and diagnostics.
- Equity: `results/equity.csv` and `results/equity.parquet`.
- Lock diagnostics: `results/lock_timeline.csv`.
- Sweeps: `results/leaderboard_guarded.csv` and per-variant artifacts under `results/sweeps/`.
