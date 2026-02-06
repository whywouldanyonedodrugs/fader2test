# STATE (single source of truth)

Repo: whywouldanyonedodrugs/fader2test
Working dir (machine): /opt/fader2

## Phase
Phase 2 In Progress (Short-Only Refactor).
Ticket 2.1 complete: short execution engine enabled; signals temporarily tagged `side="short"` for plumbing verification.

## Non-negotiables
- No-guessing policy: if a fact/semantics is unknown, stop and require the source (code/docs/output).
- Upstream parity is sacred: upstream_longonly behavior must not change.
- Local canary gate required: `tools/canary_all.sh` must PASS before any merge.
- CI is advisory: The self-hosted runner mirrors the local gate, but local verification is still mandatory.

## Canary definition
Dataset:
- Source lake: /opt/testerdonch/parquet (local; not committed)
- Symbols: symbols_canary.txt (10 symbols)
- Window: 2024-01-01T00:00:00Z .. 2024-06-01T00:00:00Z

Runners:
- Primary: tools/canary_all.sh (Runs parity + baseline check)
- Underlying: tools/check_canary.py

## Recorded baselines (committed)
- Docs/baselines/BASELINE_UPSTREAM_CANARY.json
  - signals_rows=457, trades_rows=59 (primary anchor)

## Current status
- Parity: Expected FAIL (short engine enabled; legacy signals are tagged short for testing)
- Baseline: Upstream still the reference; run baseline check directly when parity is expected to fail
- CI: Configured on self-hosted runner (.github/workflows/canary.yml)

## Next Tickets (Phase 2)
1) Ticket 2.2: Delete Donchian logic and implement Boom/Stall/Trigger signal generation.
2) Create a new shortonly baseline once the new strategy is intentional and stable.

## How to run the gate (local)
bash tools/canary_all.sh
