# Current Plan (Long -> Short)

This file tracks practical next steps from the current repository state.

## Completed foundations
- Upstream snapshot imported to `upstream_longonly/` and treated as frozen reference.
- Canary tooling exists:
  - `tools/check_canary.py`
  - `tools/canary_all.sh`
- Short-only sweep stack exists in `shortonly/`:
  - `scout.py`
  - `backtester.py`
  - `sweep_mr_short_params_guarded.py`

## Near-term priorities
1. Run long full-universe sweeps in tmux on the server and archive logs/results.
2. Review sweep survivors and lock a shortlist of robust parameter families.
3. Strengthen test coverage beyond canary (unit/integration smoke checks around scout/backtester edges).
4. Decide and document baseline policy for intentional short-only behavior changes.

## Safety rails
- Do not edit `upstream_longonly/` unless fixing a proven upstream defect.
- Keep baseline updates explicit and versioned in `Docs/baselines/`.
- Require local canary evidence before merge.
