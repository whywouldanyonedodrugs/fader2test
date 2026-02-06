from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    so = repo / "shortonly"

    sys.path.insert(0, str(so))

    import config as cfg  # noqa

    # Runtime overrides for CANARY runs (no symlinks required)
    cfg.PARQUET_DIR = repo / "data_canary"
    cfg.SYMBOLS_FILE = repo / "symbols_canary.txt"
    cfg.SIGNALS_DIR = repo / "reports" / "shortonly_signals"
    cfg.RESULTS_DIR = repo / "reports" / "shortonly_results"

    # Disable meta for baseline parity
    cfg.BT_META_ONLINE_ENABLED = False
    cfg.META_SIZING_ENABLED = False
    cfg.META_PROB_THRESHOLD = None
    cfg.META_STRICT_SCHEMA = False

    for p in (cfg.SIGNALS_DIR, cfg.RESULTS_DIR):
        Path(p).mkdir(parents=True, exist_ok=True)

    import scout  # noqa
    import backtester  # noqa

    n = scout.run_scout()
    print(f"[canary] scout signals: {n}")

    if hasattr(backtester, "main"):
        backtester.main()
    else:
        backtester.run_backtest()

    print("[canary] backtester done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
