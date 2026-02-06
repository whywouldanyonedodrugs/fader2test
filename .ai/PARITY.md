# Parity & regression policy (long-only upstream -> short-only refactor)

## Goal
Prevent silent drift while refactoring proven long-only mechanics into a short-only system.

## Two parity phases
Phase 1 — Characterization parity (required before short work)
- The refactored engine must reproduce upstream long-only outputs on canary (within defined tolerances).
- This proves the refactor did not change mechanics accidentally.

Phase 2 — Intentional behavior change (short-only)
- Once short-only is implemented, outputs WILL differ from upstream.
- At that point, establish a new golden baseline and lock it with regression gates.

## Tiers
Tier A — unit/invariant tests
Tier B — canary regression
Tier C — full-lake regression (server-only)

## Golden baselines
Maintain separate golden sets:
- `golden/upstream_longonly/` (characterization)
- `golden/fader2_shortonly/` (new target behavior)

Record sha256 for key artifacts:
- trades.parquet, metrics.json, config snapshot, data_canary MANIFEST.

## Strict rule
If outputs change:
- either fix the regression (unintended drift),
- or document and update the relevant golden baseline (intended change).
