from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _hash_dataframe_stable(df: pd.DataFrame) -> str:
    """
    Deterministic-ish fingerprint for a small canary DF on a fixed environment.
    We sort rows/cols and hash pandas' row-hash output.
    """
    if df is None or df.empty:
        return "empty"

    # normalize column order for stability
    cols = sorted(df.columns.tolist())
    df2 = df[cols].copy()

    # normalize timestamp column if present
    if "timestamp" in df2.columns:
        df2["timestamp"] = pd.to_datetime(df2["timestamp"], utc=True, errors="coerce")

    # sort rows if keys exist
    sort_keys = [c for c in ["timestamp", "symbol"] if c in df2.columns]
    if sort_keys:
        df2 = df2.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    else:
        df2 = df2.reset_index(drop=True)

    # row-wise stable hash
    row_hash = pd.util.hash_pandas_object(df2, index=False).to_numpy(dtype="uint64")
    h = hashlib.sha256(row_hash.tobytes())
    return h.hexdigest()


def _load_signals(signals_dir: Path) -> pd.DataFrame:
    parts = sorted(signals_dir.rglob("part-*.parquet"))
    if not parts:
        return pd.DataFrame()
    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return df


def _find_first(existing: List[Path]) -> Optional[Path]:
    for p in existing:
        if p.exists() and p.is_file():
            return p
    return None


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    signals_dir = repo / "reports" / "upstream_signals"
    results_dir = repo / "reports" / "upstream_results"

    out: Dict[str, Any] = {
        "repo": str(repo),
        "signals_dir": str(signals_dir),
        "results_dir": str(results_dir),
        "signals": {},
        "trades": {},
        "results_files": {},
    }

    # Signals fingerprint
    sig_df = _load_signals(signals_dir)
    out["signals"]["n_rows"] = int(len(sig_df))
    if "timestamp" in sig_df.columns:
        ts = pd.to_datetime(sig_df["timestamp"], utc=True, errors="coerce")
        out["signals"]["ts_min"] = None if ts.isna().all() else str(ts.min())
        out["signals"]["ts_max"] = None if ts.isna().all() else str(ts.max())
    out["signals"]["fingerprint_sha256"] = _hash_dataframe_stable(sig_df)

    # Trades (try common filenames; keep flexible)
    candidates = [
        results_dir / "trades.parquet",
        results_dir / "trades.csv",
        results_dir / "results" / "trades.parquet",
        results_dir / "results" / "trades.csv",
    ]
    trades_path = _find_first(candidates)
    if trades_path is not None:
        if trades_path.suffix == ".parquet":
            tdf = pd.read_parquet(trades_path)
        else:
            tdf = pd.read_csv(trades_path)
        out["trades"]["path"] = str(trades_path)
        out["trades"]["n_rows"] = int(len(tdf))
        if "entry_ts" in tdf.columns:
            ets = pd.to_datetime(tdf["entry_ts"], utc=True, errors="coerce")
            out["trades"]["entry_min"] = None if ets.isna().all() else str(ets.min())
            out["trades"]["entry_max"] = None if ets.isna().all() else str(ets.max())
        out["trades"]["fingerprint_sha256"] = _hash_dataframe_stable(tdf)
    else:
        out["trades"]["path"] = None
        out["trades"]["n_rows"] = 0
        out["trades"]["fingerprint_sha256"] = "missing"

    # Hash every result file (small directory, canary-safe)
    for p in sorted(results_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(results_dir))
            out["results_files"][rel] = {
                "bytes": p.stat().st_size,
                "sha256": _sha256_file(p),
            }

    snap_path = repo / "Docs" / "baselines" / "BASELINE_UPSTREAM_CANARY.json"
    snap_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print("Wrote:", snap_path)
    print("signals_rows:", out["signals"]["n_rows"], "signals_fp:", out["signals"]["fingerprint_sha256"])
    print("trades_rows:", out["trades"]["n_rows"], "trades_fp:", out["trades"]["fingerprint_sha256"])
    print("results_files:", len(out["results_files"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
