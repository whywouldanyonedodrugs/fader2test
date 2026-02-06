# Baselines

Baselines are committed “expected outputs” for deterministic canary runs.

- `BASELINE_UPSTREAM_CANARY.json`
  - Canonical upstream behavior for the pinned canary window and symbols.
  - If this changes, it must be a deliberate upstream snapshot change.

Optional future baselines (shortonly milestones):
- `BASELINE_SHORTONLY_M0_PARITY.json` (shortonly == upstream, refactor-only phase)
- `BASELINE_SHORTONLY_M1_STRATEGY.json` (first intentional behavior change)
