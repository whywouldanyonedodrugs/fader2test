# sweep_mr_short_params_guarded.py
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path
from typing import Optional, Tuple
import numpy as np, pandas as pd

import config as cfg
from scout import run_scout
from backtester import run_backtest
from sweep_policy_settings_v2 import ARCS, build_variants

RESULTS = Path(cfg.RESULTS_DIR)
SIGNALS = Path(cfg.SIGNALS_DIR)
RESULTS.mkdir(parents=True, exist_ok=True)
SIGNALS.mkdir(parents=True, exist_ok=True)

# ------------------ helpers to set config ------------------

def apply_entry_cfg(cfg_module, variant: dict):
    """
    Apply entry/strategy variant settings directly onto cfg.
    V2 variants already use the correct config keys (BOOM_WIN, STALL_WIN, etc).
    """
    for k, v in variant.items():
        setattr(cfg_module, k, v)

    # Ensure short mode if config supports it
    if hasattr(cfg_module, "TRADE_SIDE"):
        setattr(cfg_module, "TRADE_SIDE", "short")

def set_exit_cfg(sl_atr: float, tp_atr: float, te_hours: Optional[float]):
    cfg.SL_ATR_MULT = float(sl_atr)
    cfg.TP_ATR_MULT = float(tp_atr)
    cfg.TIME_EXIT_HOURS = float(te_hours) if te_hours is not None else None

def set_partial_trail_cfg(partial_enabled: bool,
                          partial_ratio: Optional[float],
                          tp1_atr_mult: Optional[float],
                          trail_after_tp1: bool,
                          trail_atr_mult: Optional[float]):
    # Partials
    cfg.PARTIAL_TP_ENABLED = bool(partial_enabled)
    if partial_enabled:
        cfg.PARTIAL_TP_RATIO = float(partial_ratio if partial_ratio is not None else getattr(cfg, "PARTIAL_TP_RATIO", 0.5))
        cfg.PARTIAL_TP1_ATR_MULT = float(tp1_atr_mult if tp1_atr_mult is not None else getattr(cfg, "PARTIAL_TP1_ATR_MULT", 5.0))
    # Trailing
    cfg.TRAIL_AFTER_TP1 = bool(trail_after_tp1)
    if trail_after_tp1:
        cfg.TRAIL_ATR_MULT = float(trail_atr_mult if trail_atr_mult is not None else getattr(cfg, "TRAIL_ATR_MULT", 1.0))

def _hash_entry_cfg(variant: dict) -> str:
    payload = dict(variant)
    payload.update({
        "START": str(cfg.START_DATE),
        "END": str(cfg.END_DATE),
        "ATR_TIMEFRAME": str(getattr(cfg, "ATR_TIMEFRAME", "1h")),
        "REGIME_TIMEFRAME": str(getattr(cfg, "REGIME_TIMEFRAME", "4h")),
    })
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def ensure_signals(variant: dict) -> Path:
    """Generate (or reuse) signals for the CURRENT entry config only."""
    h = _hash_entry_cfg(variant)
    out_dir = SIGNALS / f"signals_entry_{h}"
    if out_dir.exists() and list(out_dir.rglob("*.parquet")):
        return out_dir
    print("[scout] generating signals for entry config…")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.SIGNALS_DIR = out_dir
    n = run_scout()
    print(f"[scout] wrote {int(n)} signals → {out_dir}")
    return out_dir

# ------------------ metrics & robustness proxies ------------------

def _safe_read_csv(p: Path, **kwargs) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, **kwargs)
    except Exception:
        return pd.DataFrame()

def metrics_from_trades_and_equity(trades_csv: Path, equity_csv: Path) -> dict:
    out = {"n": 0, "win_rate": np.nan, "profit_factor": np.nan, "avg_R": np.nan,
           "max_dd": np.nan, "cagr": np.nan, "mar": np.nan, "daily_sharpe": np.nan,
           "min_block_sharpe": np.nan}

    # ----- Trades-based metrics (R-aware if available) -----
    try:
        tdf = pd.read_csv(trades_csv)
    except Exception:
        tdf = pd.DataFrame()

    if not tdf.empty:
        if "pnl_R" in tdf.columns:
            r = pd.to_numeric(tdf["pnl_R"], errors="coerce").dropna()
            out["n"] = int(len(r))
            out["win_rate"] = float((r > 0).mean()) if len(r) else np.nan
            pos = r[r > 0].sum()
            neg = -r[r < 0].sum()
            out["profit_factor"] = float(pos / neg) if neg > 0 else (float("inf") if pos > 0 else np.nan)
            out["avg_R"] = float(r.mean()) if len(r) else np.nan
        else:
            out["n"] = int(len(tdf))
            if {"entry", "exit"}.issubset(tdf.columns):
                out["win_rate"] = float((tdf["exit"] > tdf["entry"]).mean())

    # ----- Equity-based risk metrics -----
    try:
        edf = pd.read_csv(equity_csv, parse_dates=["timestamp"])
    except Exception:
        edf = pd.DataFrame()

    if not edf.empty and "equity" in edf.columns:
        edf = edf.dropna(subset=["timestamp", "equity"]).copy()
        edf["timestamp"] = pd.to_datetime(edf["timestamp"], utc=True, errors="coerce")
        edf = edf.dropna(subset=["timestamp"]).sort_values("timestamp")

        # Deduplicate timestamps (keep last) to avoid asfreq() errors
        edf = edf[~edf["timestamp"].duplicated(keep="last")]
        edf = edf.set_index("timestamp")

        eq = pd.to_numeric(edf["equity"], errors="coerce").dropna()
        if len(eq) >= 3:
            # Max drawdown (from the full-resolution curve)
            peak = np.maximum.accumulate(eq.values)
            dd = (eq.values / peak - 1.0).min()  # negative
            max_dd = float(-dd)
            out["max_dd"] = max_dd

            # CAGR using first/last timestamps
            t0, t1 = eq.index[0], eq.index[-1]
            years = max((t1 - t0).total_seconds() / (365.25 * 24 * 3600), 1e-9)
            cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if eq.iloc[0] > 0 else np.nan
            out["cagr"] = cagr
            out["mar"] = float(cagr / max_dd) if max_dd > 0 else np.nan

            # Daily returns via resample-last (robust to duplicates/irregular spacing)
            daily_eq = eq.resample("1D").last().ffill()
            drets = daily_eq.pct_change().dropna()
            if len(drets) >= 10:
                mu, sd = float(drets.mean()), float(drets.std(ddof=1))
                out["daily_sharpe"] = (mu / sd) * np.sqrt(365) if sd > 0 else np.nan

                # Min block Sharpe across 4 equal calendar blocks
                bvals = []
                for arr in np.array_split(drets.values, 4):
                    if len(arr) >= 5:
                        m = float(np.mean(arr)); s = float(np.std(arr, ddof=1))
                        bvals.append((m / s) * np.sqrt(365) if s > 0 else np.nan)
                if bvals:
                    out["min_block_sharpe"] = float(np.nanmin(bvals))

    return out

def _tag_entry(variant: dict) -> str:
    """
    Compact tag for entry variants (short-only).
    Example: B48_S12_T0.015_A3_trg-stall_FZ1.5
    """
    def _fmt(v) -> str:
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    parts = []
    if "BOOM_WIN" in variant:
        parts.append(f"B{_fmt(variant['BOOM_WIN'])}")
    if "STALL_WIN" in variant:
        parts.append(f"S{_fmt(variant['STALL_WIN'])}")
    if "BOOM_THRESH" in variant:
        parts.append(f"T{_fmt(variant['BOOM_THRESH'])}")
    if "STALL_ATR_MAX" in variant:
        parts.append(f"A{_fmt(variant['STALL_ATR_MAX'])}")
    if "TRIGGER_TYPE" in variant:
        parts.append(f"trg-{str(variant['TRIGGER_TYPE'])}")
    if variant.get("USE_FUNDING_GATE") or variant.get("FUNDING_GATE_ENABLED"):
        parts.append("FUND")
    if "FUNDING_Z_MIN" in variant:
        parts.append(f"FZ{_fmt(variant['FUNDING_Z_MIN'])}")
    if "FUNDING_Z_THRESHOLD" in variant:
        parts.append(f"FZ{_fmt(variant['FUNDING_Z_THRESHOLD'])}")
    return "_".join(parts)

def _tag_exit(sl, tp, te, p_en, p_ratio, tp1, tr_en, tr_mult) -> str:
    te_str = "None" if (te is None) else f"{float(te):.1f}"
    p_str  = f"PT-{int(bool(p_en))}"
    if p_en:
        p_str += f"@{float(p_ratio):.2f}_TP1x{float(tp1):.1f}"
    tr_str = f"TRL-{int(bool(tr_en))}"
    if tr_en:
        tr_str += f"x{float(tr_mult):.1f}"
    return f"SL-{sl:.1f}_TP-{tp:.1f}_TE-{te_str}_{p_str}_{tr_str}"

# ------------------ main sweep ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end",   type=str, default=None)
    ap.add_argument(
        "--archetype",
        type=str,
        default="breakdown_retest_vwap",
        choices=sorted(ARCS.keys()),
        help="Which short-only archetype grid to sweep."
    )
    ap.add_argument("--single", action="store_true",
                    help="run only the first variant for a smoke-test")
    ap.add_argument("--use-1m", action="store_true",
                    help="use 1m intrabar tie resolution in sweep (slower)")
    ap.add_argument("--robust-topk", type=int, default=0,
                    help="run reporting.py (PSR/DSR + PBO) on the top-K variants after the sweep")
    args = ap.parse_args()

    if args.start: cfg.START_DATE = args.start
    if args.end:   cfg.END_DATE   = args.end

    # --- PRE-FLIGHT CHECK ---
    # Force non-streaming to ensure signals are fully materialized for caching
    cfg.SCOUT_STREAMING = False
    from shared_utils import get_symbols_from_file
    symbols = get_symbols_from_file()
    symbols_file = getattr(cfg, "SYMBOLS_FILE", "symbols.txt")
    print("\n" + "=" * 60)
    print("SWEEP CONFIGURATION CHECK")
    print(f"Timeframe:   {cfg.START_DATE} to {cfg.END_DATE}")
    print(f"Universe:    {len(symbols)} symbols (from {symbols_file})")
    print("Streaming:   DISABLED (Forced for sweep caching)")
    print("=" * 60 + "\n")

    cfg.USE_INTRABAR_1M = bool(getattr(args, "use_1m", False))

    # ------ ENTRY GRID ------
    entry_variants = build_variants(args.archetype)
    if args.single:
        entry_variants = entry_variants[:1]

    # ------ EXIT GRID ------
    sls = [2.0, 2.5]
    tps = [8.0, 12.0]
    time_exits = [None, 72.0]

    partials = [
        (False, None, None)
    ]
    trailings = [
        (False, None)
    ]

    variants = []
    for variant in entry_variants:
        apply_entry_cfg(cfg, variant)
        sig_path = ensure_signals(variant)
        if not sig_path.exists():
            continue
        ent_tag = f"{args.archetype}_{_tag_entry(variant)}"
        for sl in sls:
            for tp in tps:
                for te in time_exits:
                    for (p_en, p_ratio, tp1) in partials:
                        for (tr_en, tr_mult) in trailings:
                            if tr_en and not p_en:
                                continue
                            exit_tag = _tag_exit(sl, tp, te, p_en, p_ratio, tp1, tr_en, tr_mult)
                            tag = f"{ent_tag}_{exit_tag}"
                            variants.append((tag, sig_path, sl, tp, te, p_en, p_ratio, tp1, tr_en, tr_mult))

    if args.single and variants:
        variants = variants[:1]
        print(f"Running one variant: {variants[0][0]}")

    print(f"Total variants: {len(variants)}")

    leaderboard = []
    leaderboard_path = RESULTS / "leaderboard_guarded.csv"
    sweeps_dir = RESULTS / "sweeps"
    sweeps_dir.mkdir(parents=True, exist_ok=True)

    for (tag, sig_path, sl, tp, te, p_en, p_ratio, tp1, tr_en, tr_mult) in variants:
        set_exit_cfg(sl, tp, te)
        set_partial_trail_cfg(p_en, p_ratio, tp1, tr_en, tr_mult)

        # skip empty-signal paths gracefully
        if sig_path.is_dir():
            if not any(sig_path.rglob("*.parquet")):
                print(f"[run] {tag}\nNo signals parquet found under: {sig_path}\n")
                continue
        else:
            try:
                sig = pd.read_parquet(sig_path)
            except Exception:
                print(f"[run] {tag}\nCould not read signals file: {sig_path}\n")
                continue
            if sig.empty:
                print(f"[run] {tag}\nNo signals to backtest.\n")
                continue

        print(f"[run] {tag}")
        run_backtest(signals_path=sig_path)

        # Save per-variant artifacts
        vdir = sweeps_dir / tag
        vdir.mkdir(parents=True, exist_ok=True)
        trades_csv = RESULTS / "trades.csv"
        equity_csv = RESULTS / "equity.csv"
        if trades_csv.exists(): shutil.copy2(trades_csv, vdir / "trades.csv")
        if equity_csv.exists(): shutil.copy2(equity_csv, vdir / "equity.csv")

        # Collect metrics
        m = metrics_from_trades_and_equity(vdir / "trades.csv", vdir / "equity.csv")
        m.update({
            "tag": tag,
            "sl_atr": sl,
            "tp_atr": tp,
            "time_exit_h": te if te is not None else np.nan,
            "partial": int(bool(p_en)),
            "partial_ratio": p_ratio if p_ratio is not None else np.nan,
            "tp1_atr": tp1 if tp1 is not None else np.nan,
            "trail": int(bool(tr_en)),
            "trail_atr": tr_mult if tr_mult is not None else np.nan,
        })
        leaderboard.append(m)
        # Persist after each completed variant so long sweeps can resume safely.
        try:
            pd.DataFrame(leaderboard).to_csv(leaderboard_path, index=False)
            print(f"Saved intermediate leaderboard ({len(leaderboard)} rows) -> {leaderboard_path}")
        except Exception as e:
            print(f"Warning: Failed to save intermediate csv: {e}")

    lb = pd.DataFrame(leaderboard)
    if lb.empty:
        print("No variants produced trades. Nothing to rank.")
        return

    # Sort by PF, then AvgR, then (lower) MaxDD, then MAR, then WinRate
    sort_cols = [("profit_factor", False), ("avg_R", False), ("max_dd", True), ("mar", False), ("win_rate", False)]
    lb = lb.sort_values([c for c,_ in sort_cols], ascending=[a for _,a in sort_cols])

    lb.to_csv(leaderboard_path, index=False)
    print(f"Saved leaderboard -> {leaderboard_path}")
    with pd.option_context("display.max_colwidth", None):
        print(lb.head(15).to_string(index=False))

    # Optional: run reporting on top-K
    if args.robust_topk and args.robust_topk > 0:
        top = lb.head(int(args.robust_topk)).copy()
        print(f"\n[robust] Running reporting.py on top-{len(top)} variants…")
        for _, row in top.iterrows():
            tag = row["tag"]
            vdir = sweeps_dir / tag
            # Temporarily copy this variant's trades back to default location so reporting can see it
            src = vdir / "trades.csv"
            if not src.exists():
                print(f"[robust] skip {tag}: trades.csv missing")
                continue
            shutil.copy2(src, RESULTS / "trades.csv")
            try:
                subprocess.run(
                    ["python", "reporting.py", "--run-all", "--returns-col", "pnl_R",
                     "--variant-cols", "pullback_type", "entry_rule", "don_break_len", "regime_up"],
                    check=False
                )
                # Optionally, copy any reporting outputs/logs to the variant folder
                # (your reporting script mostly prints; if it saves files, add copies here)
            except Exception as e:
                print(f"[robust] reporting failed for {tag}: {e}")

if __name__ == "__main__":
    main()
