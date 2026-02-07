# Repo map (living)

## Code zones
- upstream_longonly/  (imported snapshot; do not edit)
- shortonly/          (active short-only strategy + sweeps)
- fader2/             (reserved/legacy folder; currently not the active runtime path)
- tools/              (utilities)
- tests/              (Tier A + characterization + regression)

## Required docs
- Docs/bull_longonly_manual.md  (verbatim)
- Docs/contracts_longonly.md    (extracted data/execution contracts)
- Docs/short_only_spec.md       (explicit short-only semantics)
- Docs/meta_model_contract.md   (meta gating/sizing contract)

## Entry points
Upstream canary:
- `tools/run_upstream_canary.py`
- `upstream_longonly/scout.py`
- `upstream_longonly/backtester.py`

Short-only runtime:
- `shortonly/sweep_mr_short_params_guarded.py`
- `shortonly/scout.py`
- `shortonly/backtester.py`

Canary gate:
- `tools/canary_all.sh`
- `tools/check_canary.py`

## Tests / verification
- Canary parity and baseline checks are implemented via `tools/check_canary.py`.
- `tests/` is currently minimal; most regression coverage is canary-based at present.
