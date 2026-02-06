#!/usr/bin/env python3
"""
Create a small committed canary dataset from a large parquet lake.

Design goals:
- Deterministic slicing by symbol + [start, end)
- Preserve original columns as-is (no feature engineering here)
- Write a MANIFEST.json with per-file sha256 to support regression gates
- Avoid guessing timestamp units: support --ts-unit {auto,s,ms,us,ns}

Usage example:
python tools/make_canary_from_lake.py \
  --lake-dir parquet \
  --out-dir data_canary \
  --symbols BTCUSDT ETHUSDT \
  --start "2024-01-01T00:00:00Z" \
  --end   "2024-02-01T00:00:00Z" \
  --ts-unit auto
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd


TS_CANDIDATES = ("timestamp", "ts", "time", "datetime", "open_time", "close_time")


def _parse_iso_utc(s: str) -> pd.Timestamp:
    # pd.Timestamp handles Z; ensure tz-aware UTC
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _detect_ts_col(df, ts_col: str | None) -> str:
    """
    Determine where timestamps live.
    Returns:
      - a column name, OR
      - the special token "index" meaning: use df.index
    """

    # Explicit override
    if ts_col is not None:
        if ts_col.lower() == "index":
            if isinstance(df.index, pd.DatetimeIndex):
                return "index"
            raise ValueError(f"--ts-col index requested but df.index is not a DatetimeIndex (got {type(df.index)})")
        if ts_col in df.columns:
            return ts_col
        raise ValueError(f"--ts-col {ts_col!r} not found. Columns={list(df.columns)} index={type(df.index)}")

    # Auto-detect: prefer datetime index if present
    if isinstance(df.index, pd.DatetimeIndex):
        return "index"

    # Auto-detect: look for common timestamp column names
    candidates = ("timestamp", "ts", "time", "datetime", "open_time", "close_time", "__index_level_0__", "index")
    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        "Could not detect timestamp column. Provide --ts-col. "
        f"Looked for {candidates}. Columns={list(df.columns)} index={type(df.index)}"
    )


def _normalize_ts_series(s: pd.Series, ts_unit: str) -> pd.DatetimeIndex:
    """
    Convert a timestamp-like Series into tz-aware UTC datetime index.

    Supports:
    - datetime-like strings
    - pandas datetime
    - integer epochs (unit set by --ts-unit or auto heuristic)
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        ts = pd.to_datetime(s, utc=True)
        return ts

    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        ts = pd.to_datetime(s, utc=True, errors="raise")
        return ts

    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        vals = s.astype("int64")

        unit = ts_unit
        if unit == "auto":
            # Heuristic based on magnitude; user can override.
            v = int(vals.iloc[0])
            # seconds ~ 1e9..1e10, ms ~ 1e12..1e13, us ~ 1e15.., ns ~ 1e18..
            if v >= 10**17:
                unit = "ns"
            elif v >= 10**14:
                unit = "us"
            elif v >= 10**11:
                unit = "ms"
            else:
                unit = "s"

        ts = pd.to_datetime(vals, utc=True, unit=unit)
        return ts

    raise TypeError(f"Unsupported timestamp dtype: {s.dtype}")


def _list_symbol_files(lake_dir: Path, symbol: str) -> list[Path]:
    # Common patterns: parquet/<SYMBOL>.parquet or parquet/<SYMBOL>/*.parquet
    direct = lake_dir / f"{symbol}.parquet"
    if direct.exists():
        return [direct]
    sub = lake_dir / symbol
    if sub.exists() and sub.is_dir():
        return sorted(p for p in sub.glob("*.parquet") if p.is_file())
    # fallback: try case variations
    direct2 = lake_dir / f"{symbol.upper()}.parquet"
    if direct2.exists():
        return [direct2]
    raise FileNotFoundError(f"Could not find parquet for symbol={symbol!r} in lake_dir={lake_dir}")


@dataclass(frozen=True)
class SliceSpec:
    start: pd.Timestamp
    end: pd.Timestamp
    ts_col: Optional[str]
    ts_unit: str


def _slice_one_file(in_path: Path, out_path: Path, symbol: str, spec: SliceSpec) -> Tuple[int, str]:


    start_ts = pd.to_datetime(spec.start, utc=True) if getattr(spec, "start", None) else None
    end_ts   = pd.to_datetime(spec.end, utc=True) if getattr(spec, "end", None) else None


    df = pd.read_parquet(in_path)
    ts_col = _detect_ts_col(df, spec.ts_col)

    # ts_col may be a real column name or the special token "index"
    ts_src = df.index if ts_col == "index" else df[ts_col]
    ts = _normalize_ts_series(ts_src, spec.ts_unit)

    # Apply normalized timestamps back onto df so downstream logic is consistent
    df = df.copy()
    if ts_col == "index":
        df.index = pd.DatetimeIndex(ts, name=df.index.name or "timestamp")
    else:
        df[ts_col] = ts


    mask = (ts >= spec.start) & (ts < spec.end)
    out_df = df.loc[mask].copy()

    # --- Output normalization: ensure a physical timestamp column for downstream compatibility ---
    if isinstance(df.index, pd.DatetimeIndex):
        # Make sure the index is named 'timestamp' so reset_index produces the right column
        if (df.index.name or "") != "timestamp":
            df = df.copy()
            df.index = df.index.rename("timestamp")
        df = df.reset_index()
    else:
        # If timestamps were in a column but not named timestamp, standardize it
        if "timestamp" not in df.columns:
            # best-effort: if ts_col exists and is not timestamp, rename it
            if "ts_col" in locals() and ts_col in df.columns:
                df = df.rename(columns={ts_col: "timestamp"})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # --- Ensure output has a physical timestamp column (required by upstream schema validators) ---
    out_df = df.copy()
    if isinstance(out_df.index, pd.DatetimeIndex):
        # Put timestamp into a real column, then drop index
        out_df.insert(0, "timestamp", out_df.index)
        out_df.reset_index(drop=True, inplace=True)
    else:
        # If timestamps live in a column, standardize its name
        if "timestamp" not in out_df.columns:
            if "ts_col" in locals() and ts_col in out_df.columns:
                out_df = out_df.rename(columns={ts_col: "timestamp"})

    # --- Enforce start/end window on normalized timestamps (after normalization, before write) ---
    if start_ts is not None or end_ts is not None:
        # Determine the timestamp series/index to filter on
        if isinstance(df.index, pd.DatetimeIndex):
            ts_f = df.index
        elif "timestamp" in df.columns:
            ts_f = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        else:
            # fall back: if ts_col exists, try it
            ts_f = pd.to_datetime(df[ts_col], utc=True, errors="coerce") if "ts_col" in locals() and ts_col in df.columns else None

        if ts_f is not None:
            m = pd.Series(True, index=df.index)
            if start_ts is not None:
                m &= (ts_f >= start_ts)
            if end_ts is not None:
                m &= (ts_f <= end_ts)
            df = df.loc[m.values]



    out_df.to_parquet(out_path, index=False)


    digest = _sha256_file(out_path)
    return int(len(out_df)), digest


def _write_manifest(out_dir: Path, items: list[dict]) -> None:
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "items": items,
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lake-dir", required=True, help="Input lake dir (server-only), e.g. parquet/")
    p.add_argument("--out-dir", required=True, help="Output dir, e.g. data_canary/")
    p.add_argument("--symbols", nargs="+", required=True, help="Symbols to include, e.g. BTCUSDT ETHUSDT")
    p.add_argument("--start", required=True, help='ISO start, e.g. "2024-01-01T00:00:00Z"')
    p.add_argument("--end", required=True, help='ISO end (exclusive), e.g. "2024-02-01T00:00:00Z"')
    p.add_argument("--ts-col", default=None, help="Timestamp column name (optional). If omitted, auto-detect.")
    p.add_argument(
        "--ts-unit",
        default="auto",
        choices=("auto", "s", "ms", "us", "ns"),
        help="Epoch unit for integer timestamps. Use 'auto' or specify explicitly.",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    lake_dir = Path(args.lake_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = _parse_iso_utc(args.start)
    end = _parse_iso_utc(args.end)
    if end <= start:
        raise ValueError(f"--end must be > --start (got start={start}, end={end})")

    spec = SliceSpec(start=start, end=end, ts_col=args.ts_col, ts_unit=args.ts_unit)

    items: list[dict] = []
    for sym in args.symbols:
        files = _list_symbol_files(lake_dir, sym)
        for in_path in files:
            # Write to data_canary/<SYMBOL>.parquet (flattened).
            out_path = out_dir / f"{sym}.parquet"
            n_rows, sha = _slice_one_file(in_path, out_path, sym, spec)
            items.append(
                {
                    "symbol": sym,
                    "input": str(in_path),
                    "output": str(out_path),
                    "rows": n_rows,
                    "sha256": sha,
                    "start_utc": str(start),
                    "end_utc": str(end),
                    "ts_col": spec.ts_col,
                    "ts_unit": spec.ts_unit,
                }
            )
            # If symbol maps to multiple files, last one wins in flattened output;
            # for that case, prefer lake format with one file per symbol.
            break

    _write_manifest(out_dir, items)
    print(f"Wrote canary dataset to: {out_dir}")
    print(f"Manifest: {out_dir / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
