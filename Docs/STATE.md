# STATE (read this first)

Last updated (UTC): 2026-02-06
Phase: Workflow + canary harness complete (upstream imported; shortonly parity locked)

## Non-negotiables
- No-guessing policy: if unsure, stop and ask / inspect repo.
- Strict parity gates: any refactor must keep canary outputs identical until we intentionally change semantics.
- No intrabar 1m usage (USE_INTRABAR_1M must remain False; bt_intrabar optional).
- Upstream snapshot is “read-mostly”: minimize edits inside upstream_longonly; prefer wrappers/tools.

## Repo layout
- upstream_longonly/ : imported upstream snapshot (reference)
- shortonly/         : working copy (initially identical to upstream)
- tools/             : harness scripts (canary runner, snapshot, compare, check)
- Docs/baselines/    : baselines + workflow docs
- reports/           : local runtime outputs (gitignored)
- data_canary/       : local canary dataset (gitignored; generated from lake)

## Canary (definition of “parity passes”)
- Canary dataset: 10 symbols, 5m, 2024-01-01..2024-06-01 (generated from /opt/testerdonch/parquet)
- Commands:
  - python tools/run_upstream_canary.py
  - python tools/run_shortonly_canary.py
  - python tools/snapshot_upstream_canary.py
  - python tools/compare_upstream_shortonly_canary.py
  - python tools/check_canary.py  (gate)
- Baseline file:
  - Docs/baselines/BASELINE_UPSTREAM_CANARY.json

## Latest known-good baseline
- Baseline commit: <git sha>
- BASELINE_UPSTREAM_CANARY.json:
  - signals_rows=457
  - trades_rows=59
  - signals_fp=944598160459…
  - trades_fp=a2f328569690…
- shortonly parity status: MATCHES upstream canary (as of <git sha>)

## Current blockers / gotchas
- GitHub Actions CI cannot reproduce local canary unless we commit a tiny fixture dataset or add a download step.
- Some symbols lack open_interest/funding_rate columns; canary set includes symbols that have both where needed.

## Next 3 tickets (do next, in order)
1) Commit shortonly snapshot + provenance (so work starts from a pinned identical baseline).
2) Add Docs/baselines/BASELINE_SHORTONLY_CANARY.json (optional now, mandatory once shortonly diverges intentionally).
3) Decide CI strategy: (A) local-only gate, or (B) commit micro-canary fixtures to run in GitHub Actions.

## “Start here” reading list for any new chat/agent
- Docs/STATE.md
- Docs/baselines/CANARY_WORKFLOW.md
- .ai/PARITY.md
- .ai/COMMANDS.md
- Docs/short_only_spec.md
