# Meta-model gating/sizing contract (critical)

## Two scoring paths (must remain equivalent)
A) Offline merge-asof (meta_p from parquet)
- Required columns in prediction parquet:
- timestamp rounding:
- tolerance:
- dedup rule when multiple preds per (ts,symbol):
- merge_asof direction (backward) and why:

B) Online scoring (exported bundle)
- required artifacts:
  - model pipeline
  - feature_manifest.json (authoritative schema)
  - calibration config
  - thresholds (if gating)
  - sizing curve (if sizing)
  - checksums
- strict schema enforcement rules:
  - missing key -> hard fail (fail-closed)
  - NaNs allowed only if key present and model supports it
- calibration methods supported and expected outputs range

## Directionality rule (non-negotiable)
Do not reuse a long-signed feature name with inverted meaning under the same name.
If a feature’s semantics change for shorts, either:
- rename it, or
- define a side-invariant signed convention and enforce it everywhere (scout, backtester, online scorer, training).

## Leakage rule
Any feature used for training/gating must be past-only (no smoothed probabilities / future leakage).
