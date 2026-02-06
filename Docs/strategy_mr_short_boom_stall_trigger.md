“This is the intended short-only strategy direction; implementation must not begin until characterization parity against upstream long-only mechanics is in place.”

“Regime logic will be replaced with the long-only regime approach once imported and verified.”

TL;DR (non-quants)

1. Find perps that had a strong upside “boom” with elevated volume, then started “stalling” above an anchored VWAP (AVWAP), and short when price breaks down (or an EMA trigger flips).
    
2. Prefer setups where upside is crowded (high positive funding z-score and/or crowding rank), or where a liquidation spike likely marked exhaustion.
    
3. Take profit either back to VWAP (mean reversion) or via fixed multiples; stop is ATR-based (optionally VWAP-aware).
    
4. Search multiple archetypes (different “boom/stall/trigger” definitions + overlays) and downselect via early-stopping and robust validation (PSR/DSR/CPCV).
    
5. Core hypothesis and why the system is shaped this way  
    The system is built around two empirically common microstructure effects in perp markets:
    

- Mean reversion after sharp upside expansions: strong upside legs (“booms”) often transition into lower-volatility consolidation (“stall/coil”) and then retrace, frequently toward an anchored VWAP reference. The implementation explicitly encodes “boom” detection, “stall” compression, then a “trigger” that confirms breakdown. (/mnt/data/scout.py:303–311, 466–508)
    
- Crowding/carry decay overlays: strongly positive funding is treated as a crowding signal (longs paying shorts), which can both (a) subsidize holding shorts and (b) indicate vulnerable positioning. The project uses funding z-scores, crowding rank, and optional OI/liquidation proxies as gates layered on top of the base MR pattern. (/mnt/data/scout.py:363–395; /mnt/data/sweep_mr_short_params_guarded.py:295–348)
    

2. System architecture (how the parts connect)  
    A. Scout (signal generator): for each symbol, it loads 5m OHLCV and optional enrichments (open interest, funding_rate, liq_buy_usd, liq_sell_usd), computes ATR/ADX on a higher timeframe mapped back to 5m, computes boom return and boom z-score, volume z-score, VWAP (rolling or anchored), then iterates over time to detect “boom → stall → trigger” episodes. (/mnt/data/scout.py:313–361, 397–433, 446–508)
    

B. Backtester (trade simulator): consumes signals (parquet), applies risk sizing (percent-of-equity or fixed cash), enforces portfolio limits (notional cap, leverage), per-day trade caps and per-symbol cooldown, and simulates exits with several stop/TP styles, optional partials/trailing, and an intrabar tie-breaker (optional 1m) when SL and TP are hit within the same 5m bar. (/mnt/data/backtester.py:179–211, 403–433, 445–507)

C. Sweep runner / archetypes: defines multiple archetypes (parameter grids) and overlay gate-grids; expands each archetype into candidates (entry × exit × time-exit), and evaluates them via successive halving (shorter history + fewer symbols first, then full). It supports grid, Sobol(16), and PB12 (Plackett–Burman) screening designs. (/mnt/data/sweep_mr_short_params_guarded.py:241–259, 295–393, 397–548, 551–616, 1035–1235)

D. Reporting/robustness: summarizes sweeps, computes performance metrics, and optionally runs robust inference tools (Probabilistic Sharpe Ratio, Deflated Sharpe Ratio, CPCV with purging/embargo). (/mnt/data/reporting.py:1–80, 235–275)

3. The base signal logic (the “boom → stall → trigger” template)  
    This is the common backbone across archetypes; archetypes mostly move thresholds and add overlays.
    

3.1 Boom (first-passage “event”)

- Boom window W (in minutes, converted to bars) and stall window S are configured.
    
- Boom condition is: (return over W >= BOOM_RET_PCT OR z-score over W >= BOOM_Z_R) AND volume z-score >= BOOM_VOL_Z. The boom detection uses a first-passage style check (i.e., only count the moment the threshold is crossed, not every bar it remains above) to avoid repeated triggers during the same regime. (/mnt/data/scout.py:397–491)
    
- Episode dedup: once a boom is accepted, a cooldown prevents overlapping/adjacent booms from generating multiple signals. (/mnt/data/scout.py:453–498)
    

Rationale: first-passage + dedup converts “regime” into “event”, which is critical for universe-level scalability (otherwise a handful of trending names dominate the signal count).

3.2 VWAP context (anchoring matters)  
At the stall end, the scout computes VWAP in one of three modes:

- rolling VWAP over VWAP_ROLL_MIN,
    
- anchored_day VWAP (reset daily),
    
- anchored_boom_start VWAP (AVWAP from boom start). (/mnt/data/scout.py:420–508)
    

Many archetypes intentionally use anchored_boom_start to measure “distance above AVWAP” as a normalized overextension metric (DIST_ABOVE_VWAP_ATR_MIN, in ATR units).

3.3 Stall (compression/coil filter)  
A stall is a window after the boom where:

- range is small relative to ATR (STALL_RANGE_ATR_MAX),
    
- slope in ATR units is constrained (STALL_SLOPE_ATR_MAX),
    
- and optionally a minimum number of “flat/small” bars is required (STALL_BARS_FLAT_MIN), plus optional pattern gates (upper-wick ratio threshold, inside-bar requirement). (/mnt/data/scout.py:514+; /mnt/data/sweep_mr_short_params_guarded.py:241–259, 947–969)
    

Rationale: the stall is used as a structural “distribution” phase; it improves entry timing and creates a meaningful “stall low” level for breakdown triggers.

3.4 Cooling / “MR regime” confirmation  
The scout can apply multiple “cooling” gates (combined via COOLING_VR_COMBINE):

- Jump cooling based on bipower variation / realized variation (filters out jump-driven continuation risk),
    
- Variance ratio gate (VR close to 1, i.e., closer to random walk / mean-reverting regime),
    
- ACF gate (low autocorrelation at a specific lag), and optional RV drop. (/mnt/data/sweep_mr_short_params_guarded.py:351–393; /mnt/data/scout.py:307–309)
    

Rationale: avoid shorting into genuine momentum regimes.

3.5 Optional perp overlays (crowding / OI / liquidation)  
These are configured as gates:

- Funding z-score gate: funding_rate is converted to a rolling z-score over FUNDING_Z_LOOKBACK_H; you can require positive funding and a minimum z. (/mnt/data/scout.py:363–375)
    
- OI delta gate: require open_interest % change over OI_DELTA_LOOKBACK_H above threshold (interpreted as participation / leverage buildup). (/mnt/data/scout.py:376–385)
    
- Liquidation spike gate: prefer real liquidation feed if present, else use a proxy based on wick/volume/jumpiness. (/mnt/data/scout.py:386–395; /mnt/data/sweep_mr_short_params_guarded.py:310–319)
    

3.6 Trigger (entry timing)  
Two triggers are supported:

- break_stall_low (with an epsilon-ATR excursion threshold TRIGGER_MIN_EXCUR_ATR),
    
- ema_cross_down (fast/slow EMA cross). (/mnt/data/scout.py:310–311; /mnt/data/sweep_mr_short_params_guarded.py:947–997)
    

A “lookahead L” requirement enforces that the trigger occurs within TRIGGER_LOOKAHEAD_BARS after stall completion; otherwise, the setup is discarded as stale.

4. Archetypes (what was explored and why they differ)  
    Archetypes are defined as discrete parameter sets (mostly lists of candidate values) plus a per-archetype “overlay gate grid” (ARC_GATES) and an “entry gate preset grid” (ENTRY_GATE_GRID). (/mnt/data/sweep_mr_short_params_guarded.py:241–259, 295–393, 397–548)
    

Important operational note: the sweep CLI default archetype is "intraday_squeeze", but the current ARCS dictionary does not include it; the valid choices are ARCS.keys() plus "all". This should be cleaned up before handoff. (/mnt/data/sweep_mr_short_params_guarded.py:83–84, 397–549)

4.1 Entry-gate presets (shared across archetypes)  
ENTRY_GATE_GRID defines named presets from “strict” to “very loose” (eg_days_wide) that control VR/ACF/jump-cooling/ADX caps and how cooling gates combine (“or” vs “and”). This is intended to let the sweep trade off signal count vs purity explicitly. (/mnt/data/sweep_mr_short_params_guarded.py:351–393)

4.2 Per-archetype gate overlays (ARC_GATES)  
ARC_GATES maps friendly keys (funding_z_gate, liq_spike_gate, crowding_rank_gate, etc.) into config flags via GATE_KEY_TO_CFG. The design intent is: keep the base pattern broad, and use overlays to isolate specific hypotheses. (/mnt/data/sweep_mr_short_params_guarded.py:271–292, 295–349)

4.3 The archetypes (current set: 8)  
All use anchored_boom_start VWAP mode, but vary boom horizon, required boom return/vol shock, stall structure, trigger type, and overlays.

1. avwap_lower_high_reject  
    Broader boom horizon (12–24h), permissive boom return set, looser volume z, longer stall windows, and allows either stall-low break or EMA cross. No crowding/funding overlay by default. Intent: “bear-rally MR near AVWAP” style entries with longer holds. (/mnt/data/sweep_mr_short_params_guarded.py:398–415, 295–297)
    
2. funding_spike_crowding_fade  
    Boom horizons 2–8h and higher volume z; requires crowding overlay: funding_z_gate = True with positive funding, plus crowding_rank_gate requiring top-quintile-ish crowding (max_crowding_rank_q ~ 0.85–0.90). Intent: short crowded upside legs where carry is strongly positive and positioning is vulnerable. (/mnt/data/sweep_mr_short_params_guarded.py:417–434, 298–308)
    
3. liq_aftershock_short  
    Shorter boom horizons (1–2h), high vol z; requires liq_spike_gate = True and allows using liquidation proxies if real feeds are missing (upper wick, volume z, jump ratio thresholds). Intent: fade post-liquidation exhaustion (“aftershock”). (/mnt/data/sweep_mr_short_params_guarded.py:436–453, 310–319)
    
4. breakdown_retest_vwap  
    Medium boom horizons (3–6h), tighter stall slope cap; triggers include stall-low break and EMA cross. No overlays by default. Intent: continuation-style breakdown where VWAP acts as retest/gravity. (/mnt/data/sweep_mr_short_params_guarded.py:455–472, 333)
    
5. ofi_exhaustion_proxy  
    Boom horizons 2–3h with high vol z, tight stall range; requires liq_spike_gate True (lightly) as an exhaustion proxy. Intent: approximate “order-flow exhaustion” without true OFI by combining vol shock + wick + tight stall + liquidation hint. (/mnt/data/sweep_mr_short_params_guarded.py:474–491, 334)
    
6. rv_spike_fade  
    Boom horizons 4–8h with moderate vol z, and triggers include EMA cross or stall-low break. No overlays by default. Intent: rely on “cooling gates” (variance ratio / jump cooling / RV drop) to isolate post-RV spike fade. (/mnt/data/sweep_mr_short_params_guarded.py:493–510, 335)
    
7. avwap_lh_reject_crowded  
    Like avwap_lower_high_reject but adds crowding overlays (funding_z_gate + crowding_rank_gate). Intent: longer-horizon AVWAP rejection trades, but only when upside is crowded. (/mnt/data/sweep_mr_short_params_guarded.py:512–529, 321–331)
    
8. carry_drift_fader  
    The most “carry-native” archetype: permissive boom return (relies on carry/crowding), long boom windows up to 2 days (interpreted as “carry regime age”), and requires funding_z_gate and crowding_rank_gate. Intent: systematic shorting of names with persistently positive carry and crowded longs, expecting slow drift/decay and intermittent MR. (/mnt/data/sweep_mr_short_params_guarded.py:531–548, 337–348)
    
9. Trading and exit logic (how P&L is realized)  
    5.1 Exit parametrization  
    The sweep expands entry variants by (STOP_KIND, SL_MULT, TP_KIND, TP_R / TP_ATR_MULT, VWAP_K, TIME_EXIT_HOURS). There are per-archetype exit maps (ARC_EXIT_MAP) and time-exit lists (ARC_TIME_EXITS) with a multi-day bias for several archetypes (e.g., 24/48/72h). (/mnt/data/sweep_mr_short_params_guarded.py:551–616)
    

Stop kinds:

- ATR: SL = entry + SL_ATR_MULT_SHORT * ATR_base (short).
    
- VWAP_ATR / VWAP_BAND: stops reference VWAP with an ATR band; TP can also target VWAP + k*ATR (“to_vwap” TP). (/mnt/data/backtester.py:445–507; /mnt/data/sweep_mr_short_params_guarded.py:998–1009)
    

TP kinds:

- fixed_R: TP at R multiples of initial risk,
    
- atr_mult: TP at multiple of ATR,
    
- to_vwap: TP toward VWAP (with VWAP_ATR_K_SHORT_TP). (/mnt/data/sweep_mr_short_params_guarded.py:998–1009)
    

The backtester includes optional partial TP and trailing logic (if enabled in config), but many sweep archetype exit maps are “no partials; multi-day bias” by design. (/mnt/data/backtester.py:445–507; /mnt/data/sweep_mr_short_params_guarded.py:551–585)

5.2 Intrabar tie-breaking (SL vs TP in the same bar)  
When both TP and SL are touched inside the same 5m candle, the simulator can optionally use 1m data to decide which occurred first; otherwise it uses a deterministic tie-breaker policy (e.g., sl_wins / tp_wins / nearest-to-open). This is critical for short-horizon stops/targets and materially changes results. (/mnt/data/backtester.py:403–433)

6. Portfolio construction, sizing, and “guardrails”  
    6.1 Sizing  
    Sizing supports:
    

- RISK_MODE = "risk_pct": risk a fraction of equity per trade (RISK_PCT),
    
- RISK_MODE = "fixed_cash": fixed risk cash per trade (FIXED_RISK_CASH). (/mnt/data/config.py:139–168; /mnt/data/backtester.py:179–211)
    

Notional and leverage caps:

- NOTIONAL_CAP_PCT_OF_EQUITY limits exposure,
    
- MAX_LEVERAGE caps implied leverage. (/mnt/data/config.py:145–150; /mnt/data/backtester.py:179–211)
    

6.2 Trade throttles and abort guards  
Universe trading is controlled via:

- MAX_TRADES_PER_DAY,
    
- SYMBOL_COOLDOWN_MINUTES,
    
- plus sweep-level caps like MAX_TRADES_PER_VARIANT and abort logic based on equity drawdown fraction during a run. (/mnt/data/config.py:152–161; /mnt/data/sweep_mr_short_params_guarded.py:1035–1235)
    

6.3 Regime filter (ETH-based) <- this needs to be changed definitely, we need to use the better logic of the longonly for regimes.
A regime filter exists (toggleable) that uses ETH market data (e.g., MACD-based “regime_up”) to either block shorts entirely in up-regimes or reduce size (REGIME_SIZE_WHEN_UP). The sweep runner can optionally force regime blocking during replays. (/mnt/data/config.py:170–178; /mnt/data/sweep_mr_short_params_guarded.py:21–27)

6.4 Meta-model gating and meta-sizing  
There is an optional meta-model interface:

- signals can carry a meta probability column (default "meta_p"),
    
- a META_PROB_THRESHOLD can filter trades,
    
- META_SIZING_ENABLED can scale size by probability. (/mnt/data/config.py:181–194; /mnt/data/backtester.py:445–507)
    

7. The sweep methodology (how archetypes were explored without immediate overfitting)  
    7.1 Candidate generation  
    Each archetype defines discrete lists for boom windows, boom return thresholds, volume shock, stall windows, VWAP distance, wick/inside filters, trigger type, and lookahead. Variants are generated by:
    

- grid (build_variants),
    
- Sobol(16) sampling over a subset of knobs,
    
- PB12 screening (Plackett–Burman) for 6 two-level factors. (/mnt/data/sweep_mr_short_params_guarded.py:33–95, 96+)
    

apply_entry_cfg writes a variant’s knobs into cfg, including both entry-gate preset keys and overlay gate keys via a mapping table, so all gates are controlled uniformly by config flags. (/mnt/data/sweep_mr_short_params_guarded.py:947–989)

7.2 Successive halving (early-stopping)  
The sweep evaluates candidates under progressively larger budgets:

- smaller time window first (e.g., 60d),
    
- fewer symbols by stride subsampling (e.g., keep every 4th symbol),
    
- then longer history and full symbols for survivors. The downselect uses MAR (and then Sharpe/total return as tiebreakers). (/mnt/data/sweep_mr_short_params_guarded.py:1035–1235)
    

7.3 Robustness tooling  
Finalists can be pushed through:

- PSR / DSR to penalize selection from many trials,
    
- CPCV with purging/embargo to control leakage in time-series cross-validation. (/mnt/data/reporting.py:1–80, 235–275; /mnt/data/sweep_mr_short_params_guarded.py:65–74 in the successive-halving loop)
    

8. Practical handoff notes (what the new team should check immediately)
    
9. Fix the archetype CLI default mismatch ("intraday_squeeze" not present in ARCS). (/mnt/data/sweep_mr_short_params_guarded.py:83–84, 397–549)
    
10. Ensure the live execution interpretation matches the scout’s “lookahead” semantics: in backtests, discarding setups where the trigger doesn’t happen within L is valid if live logic is “arm the setup at stall end, enter only if trigger happens soon; else cancel.” If live trading instead triggers immediately at stall end, this becomes lookahead bias. (/mnt/data/scout.py:310–311, 400–401)
    
11. Validate data availability across the Bybit universe: funding_rate, open_interest, liquidation feeds are optional; several archetypes assume them (or proxies). Confirm coverage and whether proxies behave similarly to real liq feeds. (/mnt/data/scout.py:313–320, 363–395; /mnt/data/sweep_mr_short_params_guarded.py:310–319)
    
12. Treat intrabar handling as a first-class model choice. If using bracket-style exits, adopt 1m tie-breaking in both research and live (or make live consistent with the tie-breaker policy). (/mnt/data/backtester.py:403–433)
    
13. Decide whether regime filter should be on by default for production; the sweep runner can enforce it during replays, but sweep results may have been generated without it. (/mnt/data/sweep_mr_short_params_guarded.py:21–27)
    
14. Reconfirm portfolio-level risk: current constraints are mainly per-trade and per-day; there is no explicit cross-symbol correlation control, sector caps, or exposure netting. Add if going live across a large perp universe. (/mnt/data/config.py:145–161)
    

Referenced implementation files (for the new team to start reading in order)

1. sweep_mr_short_params_guarded.py (archetypes, gates, exits, early-stopping orchestration).
    
2. scout.py (signal detection logic and feature computation).
    
3. backtester.py (sizing, constraints, execution simulation, intrabar tie-break).
    
4. reporting.py + rebuild_summaries.py (metrics, summaries, robust stats workflows).
    
5. shared_utils.py (data loading / symbol universe utilities).