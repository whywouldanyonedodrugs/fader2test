# Meta-model gating/sizing contract (critical)

## Two scoring paths (must remain equivalent)
A) Offline merge-asof (meta_p from parquet)
- Required prediction columns (after normalization): `ts`, `sym`, `meta_p`.
- Timestamp rounding: controlled by `META_MERGE_ROUND` (default `5min`).
- Merge tolerance: controlled by `META_MERGE_TOL` (default `10min`).
- Dedup: predictions are grouped by (`ts`, `sym`) and averaged before merge.
- Merge direction: backward `merge_asof` so only prior/at-time predictions are used.

B) Online scoring (exported bundle)
- required artifacts:
  - `model.joblib`
  - `feature_manifest.json` (authoritative schema)
  - `calibration.json` (+ optional isotonic artifact when selected)
- strict schema enforcement rules:
  - missing key -> hard fail when `META_STRICT_SCHEMA=True`
  - in non-strict mode, missing keys are filled with NaN
- calibration methods:
  - `none`, `sigmoid`, `isotonic`
  - output probability must be clipped to `[0, 1]`

## Directionality rule (non-negotiable)
Do not reuse a long-signed feature name with inverted meaning under the same name.
If a feature’s semantics change for shorts, either:
- rename it, or
- define a side-invariant signed convention and enforce it everywhere (scout, backtester, online scorer, training).

## Leakage rule
Any feature used for training/gating must be past-only (no smoothed probabilities / future leakage).
