# Short-only behavioral spec (must be explicit)

This spec defines the intended short-only behavior. Do not implement until characterization parity is achieved.

## Signal (short)
- Breakout definition (daily):
- Retest definition (from below):
- Trigger bar semantics:
- Any volume/RS/liquidity gates and whether RS is inverted:

## Features (directional semantics)
Rule: directional features must be either
(A) side-specific names, or
(B) side-invariant names with a documented sign convention.

List all directional features and their meaning for shorts:
- don_dist_atr (or renamed):
- breakout level distance:
- any “both_pos” regime gates inversion:

## Execution (short)
- Entry price basis (close vs next open) and slippage model:
- SL check: bar high >= SL
- TP check: bar low <= TP
- Partial TP + trailing inversion rules:
- Intrabar resolver usage and tie-break order:

## Risk & throttles
- dedup/cooldown/max trades/day/max open positions rules:
- portfolio risk cap rules:

## Outputs
- trade schema (including side="short"):
- required metrics:
