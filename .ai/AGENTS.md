# Agents & supervisor operating procedure
## “Start here” reading list for any new chat/agent
- Docs/STATE.md
- Docs/baselines/CANARY_WORKFLOW.md
- .ai/PARITY.md
- .ai/COMMANDS.md
- Docs/short_only_spec.md



## Non-negotiable rules
- Diagnose first; no guessing. If a claim depends on code, inspect exact lines and quote them.
- All code edits must be anchored (search string + exact replacement block + surrounding context).
- Keep one canonical implementation; remove or quarantine dead paths once confirmed unused.
- Human operator runs commands on `/opt/fader2` and supplies outputs/logs.

## Roles
- Supervisor (ChatGPT): plans changes, demands evidence, enforces parity gate.
- Implementer (Codex): makes the smallest possible code edits, runs the canary loop, reports diffs.
- Operator (human): executes commands, copies outputs back into chat.

## Standard change loop
1) Read the relevant code sections (quote lines).
2) Propose minimal change + exact edit instructions.
3) Implement in `shortonly/` (not upstream) unless fixing an upstream bug required for reference to run.
4) Run:
   - upstream canary
   - shortonly canary
   - compare script
5) If mismatch: locate first divergence layer (signals → trades → equity).

## What not to do
- Don’t “fix” parity by loosening compare tolerances.
- Don’t change canary window/symbols without updating the baseline contract.
- Don’t commit `reports/` or `data_canary/`.

## Codex prompt template (implementer)
You are editing `/opt/fader2`. Only touch `shortonly/` and `tools/` unless explicitly told otherwise.
Before editing, locate and quote the exact code lines to be changed.
After editing, run the canary loop in `.ai/COMMANDS.md` and report fingerprints.
