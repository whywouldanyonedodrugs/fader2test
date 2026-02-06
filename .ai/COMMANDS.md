# Operator commands (copy/paste)

All commands are intended to run on the server in `/opt/fader2`.

## 0) Confirm repo state
cd /opt/fader2
git status
git pull

## 1) (One-time) Import upstream long-only code snapshot
Goal: copy baseline long-only modules into `upstream_longonly/` WITHOUT editing them.

Recommended: use git subtree so the snapshot is pinned in history.
(Operator must supply the upstream repo URL/path and commit hash/tag.)

Example (placeholders):
git remote add bull_upstream <BULL_REPO_URL_OR_PATH>
git fetch bull_upstream

# Import into a folder:
git subtree add --prefix upstream_longonly bull_upstream <BULL_COMMIT_OR_TAG> --squash

If subtree is not desired, fallback:
- clone bull repo next to fader2 and copy required modules into `upstream_longonly/`,
  then commit them as a single “import upstream snapshot” commit.

## 2) Environment setup
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
# choose one:
python -m pip install -e ".[dev]" || python -m pip install -e .
# or:
# python -m pip install -r requirements.txt

## 3) Tier A tests (unit/invariant; no parquet lake)
pytest -q

## 4) Build/refresh canary dataset from the lake (server only)
python tools/make_canary_from_lake.py \
  --lake-dir parquet \
  --out-dir data_canary \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --start "2024-01-01T00:00:00Z" \
  --end   "2024-02-01T00:00:00Z" \
  --ts-unit auto

## 5) Tier B: canary regression run
This must become ONE canonical command that produces `reports/canary_<stamp>/...`.

Placeholder (wire to your actual entrypoint once imported):
# python -m upstream_longonly.backtester --signals ... --out reports/canary_upstream
# python -m fader2.backtest --data data_canary --config configs/canary.yaml --out reports/canary_refactor

## 6) Evidence pack to paste back when something fails
git status
git diff
# paste the full failing command output / stack trace
# paste the path of the latest run artifacts (reports/.../)
