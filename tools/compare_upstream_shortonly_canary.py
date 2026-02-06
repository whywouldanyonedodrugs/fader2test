from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, List

import pandas as pd


def _hash_dataframe_stable(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "empty"

    cols = sorted(df.columns.tolist())
    df2 = df[cols].copy()

    # normalize common time columns
    for tc in ("timestamp", "entry_ts", "exit_ts", "ts", "time", "datetime"):
        if tc in df2.columns:
            df2[tc] = pd.to_datetime(df2[tc], utc=True, errors="coerce")

    sort_keys = [c for c in ("timestamp", "symbol", "entry_ts") if c in df2.columns]
    if sort_keys:
        df2 = df2.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    else:
        df2 = df2.reset_index(drop=True)

    row_hash = pd.util.hash_pandas_object(df2, index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(row_hash.tobytes()).hexdigest()


def _load_signals(signals_dir: Path) -> pd.DataFrame:
    parts = sorted(signals_dir.rglob("part-*.parquet"))
    if not parts:
        return pd.DataFrame()
    dfs = [pd.read_parquet(p) for p in parts]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _read_any(path_parquet: Path, path_csv: Path) -> pd.DataFrame:
    if path_parquet.exists():
        return pd.read_parquet(path_parquet)
    if path_csv.exists():
        return pd.read_csv(path_csv)
    return pd.DataFrame()


def _fingerprint_run(signals_dir: Path, results_dir: Path) -> dict:
    sig = _load_signals(signals_dir)
    trades = _read_any(results_dir / "trades.parquet", results_dir / "trades.csv")
    equity = _read_any(results_dir / "equity.parquet", results_dir / "equity.csv")
    rs = pd.read_parquet(results_dir / "rs_weekly.parquet") if (results_dir / "rs_weekly.parquet").exists() else pd.DataFrame()
    lock = pd.read_csv(results_dir / "lock_timeline.csv") if (results_dir / "lock_timeline.csv").exists() else pd.DataFrame()

    return {
        "signals_rows": int(len(sig)),
        "signals_fp": _hash_dataframe_stable(sig),
        "trades_rows": int(len(trades)),
        "trades_fp": _hash_dataframe_stable(trades),
        "equity_rows": int(len(equity)),
        "equity_fp": _hash_dataframe_stable(equity),
        "rs_rows": int(len(rs)),
        "rs_fp": _hash_dataframe_stable(rs),
        "lock_rows": int(len(lock)),
        "lock_fp": _hash_dataframe_stable(lock),
    }


def _fmt(fp: str) -> str:
    return fp if fp in ("empty", "missing") else fp[:12]


def main() -> int:
    repo = Path(__file__).resolve().parents[1]

    up_sig = repo / "reports" / "upstream_signals"
    up_res = repo / "reports" / "upstream_results"
    so_sig = repo / "reports" / "shortonly_signals"
    so_res = repo / "reports" / "shortonly_results"

    a = _fingerprint_run(up_sig, up_res)
    b = _fingerprint_run(so_sig, so_res)

    keys = [
        "signals_rows", "signals_fp",
        "trades_rows", "trades_fp",
        "equity_rows", "equity_fp",
        "rs_rows", "rs_fp",
        "lock_rows", "lock_fp",
    ]

    ok = True
    for k in keys:
        if a[k] != b[k]:
            ok = False

    print("UPSTREAM:", {k: (_fmt(a[k]) if k.endswith("_fp") else a[k]) for k in keys})
    print("SHORTONLY:", {k: (_fmt(b[k]) if k.endswith("_fp") else b[k]) for k in keys})

    if not ok:
        print("\nMISMATCHES:")
        for k in keys:
            if a[k] != b[k]:
                va = _fmt(a[k]) if k.endswith("_fp") else a[k]
                vb = _fmt(b[k]) if k.endswith("_fp") else b[k]
                print(f"  - {k}: upstream={va} shortonly={vb}")
        return 2

    print("\nOK: upstream and shortonly canary outputs match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
