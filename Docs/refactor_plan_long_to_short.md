# Refactor plan: long-only upstream -> short-only FADER2

## Constraints
- Upstream code is imported and frozen under upstream_longonly/.
- Refactor must be proven safe via characterization parity before any short-only behavior is introduced.
- Meta-model gating/sizing must remain mechanically faithful (offline merge + online scoring).

## Plan (stages)
Stage 0: Import upstream + restore missing modules
- Ensure shared_utils.load_parquet_data exists in upstream snapshot.
- Ensure bt_intrabar exists if intrabar is enabled.

Stage 1: Baseline lock
- Create canary dataset.
- Define canonical upstream canary run command.
- Save golden artifacts + hashes.

Stage 2: Refactor scaffold in fader2/
- Create package skeleton: loader, strategy, simulator, meta, regimes, reporting.
- Implement long-compat mode (for parity testing only).
- Produce identical outputs to upstream on canary.

Stage 3: Short-only implementation
- Implement lower Donch + retest-from-below signal logic.
- Implement short simulator with correct SL/TP touch semantics and slippage conventions.
- Define short-only contracts in Docs/short_only_spec.md.

Stage 4: Bear meta-model
- Train/export new meta bundle with its own feature_manifest.json.
- Ensure no long-signed feature name reuse with inverted semantics.
- Add regression tests for offline meta merge vs online scoring parity.

Stage 5: Cleanup
- Remove long-compat mode if desired.
- Keep one canonical implementation path.
