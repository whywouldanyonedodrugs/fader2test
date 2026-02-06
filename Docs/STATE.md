# STATE (single source of truth)

Phase: Workflow/Canary harness COMPLETE

Repo: whywouldanyonedodrugs/fader2test
Branch: main

Non-negotiables
- No guessing. If a fact isn’t verified from repo/files/logs, ask for the missing artifact.
- Canary gate required before/after meaningful changes:
  - python tools/run_upstream_canary.py
  - python tools/run_shortonly_canary.py
  - python tools/compare_upstream_shortonly_canary.py
  - python tools/check_canary.py (if used as a gate)
- Do not create “new versions” of STATE.md; propose minimal diffs only.

Last known good canary
- Baseline file: Docs/baselines/BASELINE_UPSTREAM_CANARY.json
- Last upstream canary run: [YYYY-MM-DD] (host: TESTMACHINE)
- Last compare result: MATCH (upstream == shortonly)

Current canary inputs
- Symbols: symbols_canary.txt (10 syms)
- Window: 2024-01-01 .. 2024-05-31
- Lake source: /opt/testerdonch/parquet (local only)

Next 3 tickets
1) [ticket]
2) [ticket]
3) [ticket]

How work is executed
- Supervisor chat: decides next ticket, defines acceptance, reviews diffs/logs.
- Codex/agent chat: writes code patches per ticket (no repo history assumptions).
- Human operator: applies patch, runs canary gates, commits.
