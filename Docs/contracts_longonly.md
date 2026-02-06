# Extracted contracts from long-only manual (ground truth)

This file is a condensed checklist of non-negotiable contracts extracted from Docs/bull_longonly_manual.md.
Only add items that are confirmed by upstream code OR are explicitly stated in the manual.

## Data inputs
- 5m parquet path pattern:
- required columns:
- optional columns:
- timestamp column/index requirements:
- UTC normalization:

## Signal dataset contract
- output layout (hive partitioning):
- required base columns:
- feature panel merge keys:
- symbol restoration behavior if partition-only:

## Trade record schema
- canonical fields (must match research pipeline expectations):
- diagnostics fields:
- regime fields:
- meta fields:

## Execution semantics (long-only baseline)
- decision timestamp semantics:
- entry fill semantics:
- SL/TP touch rules:
- intrabar tie-breaking semantics:

## Meta-model contract (baseline)
Offline merge:
- rounding:
- tolerance:
- dedup behavior:

Online scoring:
- strict schema behavior:
- required artifacts:
- calibration methods:

## Known hazards / required fixes
- daily regime leakage:
- missing dependencies:
