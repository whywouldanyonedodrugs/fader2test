# Strategy: MR Short Boom/Stall/Trigger

This document reflects the current implementation in `shortonly/` and the active sweep runner.

## Overview
- The strategy looks for upside expansion (`boom`), then compression (`stall`), then short trigger confirmation.
- It evaluates parameterized variants through guarded sweeps and ranks by realized trade/equity metrics.

## Implemented components
- Signal engine: `shortonly/scout.py`
- Execution simulator: `shortonly/backtester.py`
- Sweep orchestrator: `shortonly/sweep_mr_short_params_guarded.py`
- Archetype/policy grids: `shortonly/sweep_policy_settings_v2.py`

## Signal logic (implemented)
- Boom gate:
  - return over `BOOM_WIN` bars must exceed `BOOM_THRESH`.
- Stall gate:
  - prior `STALL_WIN` range must satisfy `range / ATR <= STALL_ATR_MAX`.
- Trigger:
  - `stall_break` (default), or
  - `ema_cross`.
- Optional funding crowding gate:
  - controlled via `USE_FUNDING_GATE` / `FUNDING_GATE_ENABLED` and funding z-score threshold.

## Execution semantics (short)
- Entry: signal row entry price.
- Stop/target construction:
  - `SL = entry + SL_ATR_MULT * ATR`
  - `TP = entry - TP_ATR_MULT * ATR`
- Touch checks:
  - stop when ask high >= SL
  - take-profit when ask low <= TP
- Supports optional:
  - partial TP (`PARTIAL_TP_ENABLED`)
  - trailing after TP1 (`TRAIL_AFTER_TP1`)
  - intrabar 1m tie-resolution (`USE_INTRABAR_1M`)

## Sweep behavior
- Sweep prints and uses effective:
  - timeframe (`START_DATE`, `END_DATE`)
  - universe size from `SYMBOLS_FILE`
- Signals are cached per entry-variant hash.
- Exit grid currently sweeps SL/TP/time-exit combinations and writes:
  - `shortonly/results/leaderboard_guarded.csv`
  - per-variant artifacts in `shortonly/results/sweeps/`

## Current default run profile
- Timeframe defaults in `shortonly/config.py`:
  - `START_DATE = "2023-01-01"`
  - `END_DATE = "2025-11-15"`
- Common archetypes used:
  - `breakdown_retest_vwap`
  - `avwap_lower_high_reject`
  - `funding_spike_crowding_fade`

## Notes
- This repo currently uses canary gates for regression safety.
- Long full-universe sweeps should be run in tmux/screen on the server due to runtime length.
