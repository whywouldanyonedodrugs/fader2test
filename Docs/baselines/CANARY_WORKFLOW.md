# Canary workflow (Phase-0 gate + Phase-1 regression)

## What a canary is
A canary is a small deterministic dataset slice + a deterministic pipeline run.
It exists to catch accidental behavior changes quickly.

Inputs:
- data_canary/*.parquet  (small fixed window slice)
- symbols_canary.txt      (fixed symbol set)

Pipeline:
1) scout (signal generation) -> reports/*_signals/
2) backtester (execution sim) -> reports/*_results/ (trades, equity, locks, rs)

Outputs we fingerprint (hash + row counts):
- signals parquet partitions (directory)
- trades.parquet
- equity.parquet
- lock_timeline.csv (if present)
- rs_weekly.parquet (if present)

This is a regression gate: “did anything change?”.

---

## Repository structure (intentional duplication)
We keep two trees on purpose:

- upstream_longonly/
  Immutable reference snapshot of the upstream long-only system.
  Only allow changes required to run in this repo (portability fixes), not logic refactors.

- shortonly/
  Active development tree. This is where strategy changes happen.

Phase-0 goal is shortonly == upstream_longonly behavior on canary.
Phase-1 goal is: shortonly diverges (by design) BUT remains stable vs a committed shortonly baseline.

---

## Phases

### Phase 0: Wiring + parity gate (completed)
Goal: shortonly outputs match upstream outputs exactly.

Gate:
- tools/check_canary.py --mode parity

Pass condition:
- Upstream snapshot == Shortonly snapshot (all fingerprints and row counts equal)

### Phase 1: Short-only development (next)
Goal: implement the short-only strategy in shortonly while retaining a regression gate.

Gate options:
A) During refactor-only steps (no intended behavior change yet):
   - tools/check_canary.py --mode parity
B) Once behavior changes are intended:
   - Create a new committed baseline for shortonly:
     Docs/baselines/BASELINE_SHORTONLY_CANARY.json
   - Use:
     tools/check_canary.py --mode baseline --baseline Docs/baselines/BASELINE_SHORTONLY_CANARY.json --target shortonly

---

## Typical operator workflow (human on the machine)

### One-time: commit upstream baseline into Docs/baselines
1) Create folder:
   mkdir -p Docs/baselines

2) Copy the already generated upstream baseline snapshot:
   cp reports/upstream_results/BASELINE_UPSTREAM_CANARY.json Docs/baselines/BASELINE_UPSTREAM_CANARY.json

3) Commit:
   git add Docs/baselines/BASELINE_UPSTREAM_CANARY.json
   git commit -m "Add upstream canary baseline"
   git push

Notes:
- data_canary/ and reports/ should NOT be committed (they are runtime artifacts).
- Only baselines (small JSON) are committed.

---

## Commands

### Regenerate canary data slice (server-only, uses the lake)
python tools/make_canary_from_lake.py \
  --lake-dir /opt/testerdonch/parquet \
  --out-dir data_canary \
  --symbols $(tr '\n' ' ' < symbols_canary.txt) \
  --start "2024-01-01T00:00:00Z" \
  --end   "2024-06-01T00:00:00Z" \
  --ts-unit auto

### Run the Phase-0 parity gate (fast regression)
python tools/check_canary.py --mode parity

### Create/refresh a shortonly baseline (only when you INTEND to lock in new behavior)
python tools/check_canary.py --mode baseline --target shortonly \
  --write-baseline Docs/baselines/BASELINE_SHORTONLY_CANARY.json

### Validate shortonly against its committed baseline
python tools/check_canary.py --mode baseline --target shortonly \
  --baseline Docs/baselines/BASELINE_SHORTONLY_CANARY.json

---

## How we use Codex safely
Supervisor chat produces a task spec with:
- exact files to touch
- exact acceptance gate (which check_canary mode must pass)
- explicit “do not modify upstream_longonly logic”

Codex implements ONLY that task, then you run:
- python tools/check_canary.py ...

If gate fails:
- revert or fix until it passes, then commit.

This prevents costly accidental drift.

Process hardening:
- Treat “local canary gate PASS + fingerprints/row counts pasted” as mandatory completion criteria for any patch.
