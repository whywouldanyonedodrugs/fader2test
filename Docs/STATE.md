# STATE (single source of truth)

Repo: whywouldanyonedodrugs/fader2test
Working dir (machine): /opt/fader2

## Phase
Workflow + canary harness complete. Upstream snapshot frozen; shortonly bootstrapped from upstream.

## Non-negotiables
- No-guessing policy: if a fact/semantics is unknown, stop and require the source (code/docs/output).
- Upstream parity is sacred: upstream_longonly behavior must not change after baseline is recorded.
- No 1m intrabar data: intrabar modules must be optional; USE_INTRABAR_1M=False.

## Canary definition
Dataset:
- Source lake: /opt/testerdonch/parquet (local; not committed)
- Symbols: symbols_canary.txt (10 symbols)
- Window: 2024-01-01T00:00:00Z .. 2024-06-01T00:00:00Z
Outputs (local; not committed): reports/upstream_* and reports/shortonly_*

Runners:
- tools/run_upstream_canary.py
- tools/run_shortonly_canary.py
- tools/compare_upstream_shortonly_canary.py
- tools/snapshot_upstream_canary.py
Gate:
- tools/check_canary.py (must pass before merge)

## Recorded baselines (committed)
- Docs/baselines/BASELINE_UPSTREAM_CANARY.json
  - signals_rows=457, trades_rows=59 (see file for fingerprints)

## Current status (last known good)
- Upstream canary: PASS
- Shortonly canary vs upstream: MATCH (fingerprints identical)
- Baseline snapshot written: reports/upstream_results/BASELINE_UPSTREAM_CANARY.json then copied to Docs/baselines/

## Next 3 tickets
1) Commit Docs/STATE.md (this file) and keep it updated on every meaningful merge.
2) Decide execution model:
   - Supervisor chat defines tickets + acceptance.
   - Codex executes edits + runs local canary + updates STATE.md.
3) Decide CI approach:
   - Local-only canary gate (default), OR
   - Self-hosted GitHub Actions runner to execute canaries on the data machine.

## How to run the gate (local)
python tools/run_upstream_canary.py
python tools/run_shortonly_canary.py
python tools/compare_upstream_shortonly_canary.py
python tools/check_canary.py --baseline Docs/baselines/BASELINE_UPSTREAM_CANARY.json
