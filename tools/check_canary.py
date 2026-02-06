#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# -----------------------------
# Helpers: hashing + row counts
# -----------------------------

def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _count_rows_parquet(path: Path) -> int:
    # Fast count via pyarrow metadata if available
    try:
        import pyarrow.parquet as pq  # type: ignore
        pf = pq.ParquetFile(str(path))
        md = pf.metadata
        return int(md.num_rows) if md is not None else 0
    except Exception:
        # Fallback: pandas read (slower)
        import pandas as pd  # type: ignore
        df = pd.read_parquet(path)
        return int(len(df))


def _count_rows_csv(path: Path) -> int:
    # Count non-header rows
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        first = True
        for _line in f:
            if first:
                first = False
                continue
            n += 1
    return n


def _hash_dataframe_rowset(df: "pd.DataFrame") -> str:
    import pandas as pd  # type: ignore
    if df is None:
        df = pd.DataFrame()
    cols = sorted(df.columns)
    df2 = df[cols]
    row_hash = pd.util.hash_pandas_object(df2, index=False).to_numpy(dtype="uint64")
    row_hash.sort()
    h = hashlib.sha256(row_hash.tobytes())
    return h.hexdigest()


def _hash_parquet_dir(dir_path: Path) -> Tuple[int, str]:
    """
    Order-insensitive row-set fingerprint for a parquet directory:
    - load all parquet parts into a DataFrame
    - canonicalize column order
    - hash the sorted row-hash vector
    """
    import pandas as pd  # type: ignore
    parts = sorted(p for p in dir_path.rglob("*.parquet") if p.is_file())
    if not parts:
        df = pd.DataFrame()
    else:
        dfs = [pd.read_parquet(p) for p in parts]
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    rows = int(len(df))
    return rows, _hash_dataframe_rowset(df)


def _hash_parquet_file_rowset(path: Path) -> Tuple[int, Optional[str]]:
    if not path.exists():
        return 0, None
    import pandas as pd  # type: ignore
    df = pd.read_parquet(path)
    return int(len(df)), _hash_dataframe_rowset(df)


def _hash_optional_file(path: Path, kind: str) -> Tuple[int, Optional[str]]:
    if not path.exists():
        return 0, None
    if kind == "parquet":
        return _count_rows_parquet(path), _sha256_file(path)
    if kind == "csv":
        return _count_rows_csv(path), _sha256_file(path)
    raise ValueError(f"Unknown kind={kind!r}")


def _short(fp: Optional[str]) -> Optional[str]:
    return fp[:12] if fp else None


# -----------------------------
# Snapshot model
# -----------------------------

@dataclass(frozen=True)
class Snapshot:
    signals_rows: int
    signals_fp: str

    trades_rows: int
    trades_fp: Optional[str]

    equity_rows: int
    equity_fp: Optional[str]

    rs_rows: int
    rs_fp: Optional[str]

    lock_rows: int
    lock_fp: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signals_rows": self.signals_rows,
            "signals_fp": self.signals_fp,
            "trades_rows": self.trades_rows,
            "trades_fp": self.trades_fp,
            "equity_rows": self.equity_rows,
            "equity_fp": self.equity_fp,
            "rs_rows": self.rs_rows,
            "rs_fp": self.rs_fp,
            "lock_rows": self.lock_rows,
            "lock_fp": self.lock_fp,
        }


def snapshot_outputs(signals_dir: Path, results_dir: Path) -> Snapshot:
    sig_rows, sig_fp = _hash_parquet_dir(signals_dir)

    trades_rows, trades_fp = _hash_optional_file(results_dir / "trades.parquet", "parquet")
    equity_rows, equity_fp = _hash_optional_file(results_dir / "equity.parquet", "parquet")
    rs_rows, rs_fp = _hash_parquet_file_rowset(results_dir / "rs_weekly.parquet")

    # lock_timeline is csv in your current upstream
    lock_rows, lock_fp = _hash_optional_file(results_dir / "lock_timeline.csv", "csv")

    return Snapshot(
        signals_rows=sig_rows,
        signals_fp=sig_fp,
        trades_rows=trades_rows,
        trades_fp=trades_fp,
        equity_rows=equity_rows,
        equity_fp=equity_fp,
        rs_rows=rs_rows,
        rs_fp=rs_fp,
        lock_rows=lock_rows,
        lock_fp=lock_fp,
    )


def diff_snap(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    keys = sorted(set(a) | set(b))
    lines = []
    for k in keys:
        av = a.get(k)
        bv = b.get(k)
        if av != bv:
            lines.append(f"- {k}: {av} != {bv}")
    return "\n".join(lines)


# -----------------------------
# Subprocess runner
# -----------------------------

def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


# -----------------------------
# Main
# -----------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Canary gate runner")
    p.add_argument(
        "--mode",
        choices=("parity", "baseline"),
        default="parity",
        help="parity: upstream vs shortonly must match. baseline: compare target vs baseline JSON.",
    )
    p.add_argument(
        "--target",
        choices=("upstream", "shortonly"),
        default="shortonly",
        help="When --mode=baseline, which side to compare to baseline.",
    )

    # Where canary data lives (runtime, not committed)
    p.add_argument("--parquet-dir", default="data_canary", help="Canary parquet directory (default: data_canary)")
    p.add_argument("--symbols-file", default="symbols_canary.txt", help="Symbols list file (default: symbols_canary.txt)")

    # Where runtime outputs go (ignored by git)
    p.add_argument("--reports-dir", default="reports", help="Reports root (default: reports/)")

    # Baseline handling
    p.add_argument(
        "--baseline",
        default="Docs/baselines/BASELINE_UPSTREAM_CANARY.json",
        help="Baseline JSON path for --mode=baseline (or reference in docs).",
    )
    p.add_argument(
        "--write-baseline",
        default=None,
        help="If set, write the chosen snapshot to this path (use only to intentionally update baselines).",
    )

    # Optional: regenerate canary from lake
    p.add_argument("--regen", action="store_true", help="Regenerate canary from lake using tools/make_canary_from_lake.py")
    p.add_argument("--lake-dir", default=None, help="Lake dir for --regen (e.g., /opt/testerdonch/parquet)")
    p.add_argument("--start", default="2024-01-01T00:00:00Z", help="Start time for --regen")
    p.add_argument("--end", default="2024-06-01T00:00:00Z", help="End time for --regen")
    p.add_argument("--ts-unit", default="auto", choices=("auto", "s", "ms", "us", "ns"), help="Timestamp unit for --regen")

    # Execution
    p.add_argument("--clean", action="store_true", help="Remove prior canary reports before running")
    args = p.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    py = sys.executable

    parquet_dir = (repo / args.parquet_dir).resolve()
    symbols_file = (repo / args.symbols_file).resolve()
    reports_root = (repo / args.reports_dir).resolve()

    # Optional regen
    if args.regen:
        if not args.lake_dir:
            raise SystemExit("--regen requires --lake-dir")
        lake_dir = Path(args.lake_dir).resolve()

        # Read symbols from symbols_file (one per line) to avoid shell quoting issues
        if not symbols_file.exists():
            raise SystemExit(f"symbols file not found: {symbols_file}")
        syms = [ln.strip() for ln in symbols_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not syms:
            raise SystemExit(f"no symbols in: {symbols_file}")

        cmd = [
            py, "tools/make_canary_from_lake.py",
            "--lake-dir", str(lake_dir),
            "--out-dir", str(parquet_dir),
            "--symbols", *syms,
            "--start", args.start,
            "--end", args.end,
            "--ts-unit", args.ts_unit,
        ]
        run(cmd, cwd=repo)

    # Ensure inputs exist
    if not parquet_dir.exists():
        raise SystemExit(f"parquet dir not found: {parquet_dir}")
    if not symbols_file.exists():
        raise SystemExit(f"symbols file not found: {symbols_file}")

    # Clean reports
    if args.clean and reports_root.exists():
        # Only remove known canary subdirs, not the whole reports tree
        for sub in (
            "upstream_signals", "upstream_results",
            "shortonly_signals", "shortonly_results",
        ):
            d = reports_root / sub
            if d.exists():
                for child in d.iterdir():
                    if child.is_dir():
                        for pth in sorted(child.rglob("*"), reverse=True):
                            if pth.is_file():
                                pth.unlink(missing_ok=True)
                            else:
                                pth.rmdir()
                        child.rmdir()
                    else:
                        child.unlink(missing_ok=True)

    # Run upstream + shortonly canaries (these scripts already do runtime config overrides)
    run([py, "tools/run_upstream_canary.py"], cwd=repo)
    run([py, "tools/run_shortonly_canary.py"], cwd=repo)

    # Compute snapshots
    up_sig = reports_root / "upstream_signals"
    up_res = reports_root / "upstream_results"
    sh_sig = reports_root / "shortonly_signals"
    sh_res = reports_root / "shortonly_results"

    up = snapshot_outputs(up_sig, up_res).to_dict()
    sh = snapshot_outputs(sh_sig, sh_res).to_dict()

    print("UPSTREAM:", {k: (_short(v) if isinstance(v, str) else v) for k, v in up.items()})
    print("SHORTONLY:", {k: (_short(v) if isinstance(v, str) else v) for k, v in sh.items()})

    if args.mode == "parity":
        if up != sh:
            print("FAIL: upstream and shortonly canary outputs do NOT match.")
            print(diff_snap(up, sh))
            return 2
        print("OK: upstream and shortonly canary outputs match.")
        return 0

    # baseline mode
    baseline_path = (repo / args.baseline).resolve()
    if not baseline_path.exists():
        raise SystemExit(f"baseline not found: {baseline_path}")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current = up if args.target == "upstream" else sh

    if args.write_baseline:
        outp = (repo / args.write_baseline).resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
        print(f"WROTE baseline: {outp}")
        return 0

    if baseline != current:
        print(f"FAIL: {args.target} snapshot does NOT match baseline: {baseline_path}")
        print(diff_snap(baseline, current))
        return 2

    print(f"OK: {args.target} snapshot matches baseline: {baseline_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
