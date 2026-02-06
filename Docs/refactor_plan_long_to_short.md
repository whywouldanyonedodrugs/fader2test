Roadmap to use in the “project supervisor” chat (no new files required)

Phase 0 (done): Repo harness

Upstream snapshot imported.

Canary generator works and produces timestamp column.

Upstream and shortonly canary match exactly.

Compare + snapshot tools exist.

Phase 1: Workflow automation and safety rails (next)

Commit baseline JSON under Docs/baselines/.

Add a single “one-command” runner (optional): tools/canary_all.sh (generates canary → upstream run → shortonly run → compare).

Add “stop-the-line” policy: any mismatch blocks merges.

Phase 2: Begin short-only refactor (later)

Make side first-class in signals and trades.

Implement short simulator (do not hack by negating prices).

Define short regime gating (risk-off logic), replacing any unsafe lookahead regime.

Meta-model: new feature manifest for short; avoid reusing long feature names with inverted meaning.