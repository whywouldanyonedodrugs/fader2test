# Canary workflow

## Unified Gate
The primary entry point for all safety checks is now:

```bash
./tools/canary_all.sh
```

This script automatically:
1. Runs the Parity Check (Upstream vs Shortonly).
2. Runs the Baseline Check (Upstream vs Committed JSON).
3. Fails fast if any step errors.

If you need to run checks separately:

```bash
./.venv/bin/python tools/check_canary.py --mode parity --clean
./.venv/bin/python tools/check_canary.py --mode baseline --target upstream --baseline Docs/baselines/BASELINE_UPSTREAM_CANARY.json
```

## CI / Automation
- A self-hosted GitHub Actions runner is configured to run `tools/canary_all.sh` on every push to `main`.
- The runner executes on the data machine, ensuring access to the local `data_canary` lake.

## Underlying Tools (Advanced Usage)
If `canary_all.sh` fails, you can debug using the individual tools:

### Run the Phase-0 parity gate (fast regression)
```bash
./.venv/bin/python tools/check_canary.py --mode parity --clean
```

### Validate against committed baseline
```bash
./.venv/bin/python tools/check_canary.py --mode baseline --target upstream --baseline Docs/baselines/BASELINE_UPSTREAM_CANARY.json
```

### Regenerate canary data slice (server-only)
```bash
./.venv/bin/python tools/make_canary_from_lake.py \
  --lake-dir /opt/testerdonch/parquet \
  --out-dir data_canary \
  --symbols $(tr '\n' ' ' < symbols_canary.txt) \
  --start "2024-01-01T00:00:00Z" \
  --end   "2024-06-01T00:00:00Z" \
  --ts-unit auto
```

## Updating a baseline intentionally
When behavior changes are intentional and approved:
```bash
./.venv/bin/python tools/check_canary.py \
  --mode baseline \
  --target shortonly \
  --write-baseline Docs/baselines/BASELINE_SHORTONLY_<TAG>.json
```
