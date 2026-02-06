# FADER2 agent operating rules (mandatory)

This repo is building a SHORT-ONLY backtester by refactoring/porting a proven LONG-ONLY system.
The long-only code does NOT originate in this repo. It will be imported as an upstream snapshot and treated as reference truth.

## Authoritative truth
1) Project files in the ChatGPT Project are authoritative and represent the repo state after git pull.
2) For any statement about code behavior: inspect the exact file section and cite the exact lines/behavior.
3) Do not infer implementation details from the manual when code is available; use the manual as contract/spec only.

## Non-negotiable hard rules
- Diagnose first; no guessing.
- When proposing code changes: provide explicit, anchored edit instructions:
  - what to search for
  - the exact block to find (include surrounding context that uniquely identifies it)
  - the exact replacement text
- Keep the codebase clean: avoid parallel implementations. Prefer one canonical implementation per feature family.
- You cannot run services here. The operator runs commands on `/opt/fader2` and provides logs/outputs.

## Refactor protocol (required)
We follow a two-track approach:

Track A — Upstream baseline (read-only)
- Import the long-only modules into `upstream_longonly/` and DO NOT EDIT THEM.
- Any changes to long-only behavior must be done in the refactored code, never by changing upstream.

Track B — Refactored short-only code
- Implement the refactor in `fader2/`.
- Early phases may run in "long-compat mode" ONLY to verify parity with upstream.
- Final target is short-only; once short-only is implemented, long-compat mode can be removed if desired.

## Regression gates (must exist before “refactor” claims)
Tier A: unit/invariant tests (no parquet lake).
Tier B: canary backtest regression (uses `data_canary/`).
Tier C: full-lake runs (server-only).

## Definition of Done (per task / PR)
A task is done only when all of these are true:
1) Diagnosis cites exact code lines (file + function + relevant snippet).
2) Minimal patch is produced (smallest change satisfying acceptance).
3) Operator commands are listed and executed:
   - Tier A: <command>
   - Tier B: <command>
4) Outputs are attached (logs + key artifacts).
5) Rollback is documented (revert commit or feature flag).

## Evidence pack (what operator pastes back on failure)
- exact command line
- full error output/trace
- `git status`
- `git diff` (changed files only)
- path to produced artifacts under `reports/...`

## Live-feasibility constraints (always keep in mind)
Even though we are backtester-focused, short-only logic must be live-feasible:
- no lookahead
- strict timestamp semantics (decision ts vs bar ts)
- bounded compute per step
- feature availability at runtime (especially meta-model features)
