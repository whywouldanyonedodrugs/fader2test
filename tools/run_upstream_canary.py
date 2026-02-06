from __future__ import annotations

import sys
from pathlib import Path

def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    up = repo / "upstream_longonly"

    sys.path.insert(0, str(up))

    import config as cfg  # noqa

    # Runtime overrides for CANARY runs (no symlinks required)
    cfg.PARQUET_DIR = repo / "data_canary"
    cfg.SYMBOLS_FILE = repo / "symbols_canary.txt"
    cfg.SIGNALS_DIR = repo / "reports" / "upstream_signals"
    cfg.RESULTS_DIR = repo / "reports" / "upstream_results"

    # Ensure dirs exist
    for p in (cfg.SIGNALS_DIR, cfg.RESULTS_DIR):
        Path(p).mkdir(parents=True, exist_ok=True)

    import scout  # noqa
    import backtester  # noqa

    n = scout.run_scout()
    print(f"[canary] scout signals: {n}")

    backtester.main() if hasattr(backtester, "main") else backtester.run_backtest()
    print("[canary] backtester done")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
