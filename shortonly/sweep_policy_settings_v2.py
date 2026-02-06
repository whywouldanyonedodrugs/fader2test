import itertools
import pandas as pd

# --- Gate Mapping ---
# Maps "friendly" keys in ARC_GATES to actual Config flags in scout.py/config.py
GATE_KEY_TO_CFG = {
    "funding_z_gate": "USE_FUNDING_GATE",
    # Add others as we implement them (e.g., liq_spike_gate)
}

# --- Archetype Gate Overlays ---
# Which gates are ENABLED by default for this archetype?
# (Can be overridden by the sweep runner)
ARC_GATES = {
    "breakdown_retest_vwap": [], # Pure price action
    "avwap_lower_high_reject": [],
    "funding_spike_crowding_fade": ["funding_z_gate"],
}

# --- Archetype Parameter Grids ---
# Defines the search space for each archetype.
# Keys must match scout.py config names (BOOM_WIN, STALL_WIN, etc.)

ARCS = {
    "breakdown_retest_vwap": {
        "BOOM_WIN": [48, 72],        # 4h, 6h
        "STALL_WIN": [12, 24],       # 1h, 2h
        "BOOM_THRESH": [0.015, 0.02], # 1.5%, 2.0%
        "STALL_ATR_MAX": [2.0, 3.0], # Tightness
        "TRIGGER_TYPE": ["stall_break", "ema_cross"],
        "FUNDING_Z_MIN": [0.0],      # Not used unless gate enabled
    },
    "avwap_lower_high_reject": {
        "BOOM_WIN": [144, 288],      # 12h, 24h
        "STALL_WIN": [24, 48],       # 2h, 4h
        "BOOM_THRESH": [0.02, 0.03],
        "STALL_ATR_MAX": [3.0, 4.0], # Looser stall
        "TRIGGER_TYPE": ["stall_break"],
        "FUNDING_Z_MIN": [0.0],
    },
    "funding_spike_crowding_fade": {
        "BOOM_WIN": [48, 96],
        "STALL_WIN": [12],
        "BOOM_THRESH": [0.015],
        "STALL_ATR_MAX": [3.0],
        "TRIGGER_TYPE": ["stall_break"],
        "FUNDING_Z_MIN": [1.5, 2.0], # Require high funding z-score (crowding)
    }
}

def build_variants(arc_name: str) -> list[dict]:
    """
    Generates a list of configuration dictionaries (variants) for the given archetype.
    Each dict contains key-value pairs to be injected into cfg.
    """
    if arc_name not in ARCS:
        raise ValueError(f"Archetype {arc_name} not found in ARCS.")

    grid = ARCS[arc_name]
    
    # Expand grid
    keys = list(grid.keys())
    values = list(grid.values())
    
    variants = []
    for combination in itertools.product(*values):
        var = dict(zip(keys, combination))
        
        # Apply Gate Logic:
        # If archetype has gates, ensure the corresponding Config Flag is True
        active_gates = ARC_GATES.get(arc_name, [])
        for g in active_gates:
            if g in GATE_KEY_TO_CFG:
                cfg_key = GATE_KEY_TO_CFG[g]
                var[cfg_key] = True
        
        variants.append(var)
        
    return variants

if __name__ == "__main__":
    # Quick test
    print(f"Variants for breakdown_retest_vwap: {len(build_variants('breakdown_retest_vwap'))}")
