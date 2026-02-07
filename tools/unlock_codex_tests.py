import re
from pathlib import Path

path = Path(".ai/AGENTS.md")
# Create if missing or update if present
if not path.exists():
    path.write_text("# AGENTS.md\n\n- **Codex Test Execution:** Codex IS ALLOWED to run read-only verification commands.\n", encoding="utf-8")
else:
    content = path.read_text(encoding="utf-8")
    
    # Add the permission rule if not there
    new_rule = "\n- **Codex Test Execution:** Codex IS ALLOWED to run read-only verification commands (e.g., unit tests, dry-run sweeps, ls, cat) to verify work. Significant mutations still require Operator confirmation.\n"
    
    if "Codex Test Execution" not in content:
        if "## Rules" in content:
            content = content.replace("## Rules", "## Rules" + new_rule)
        else:
            content += new_rule
            
    # Relax the restrictive logging requirement
    content = content.replace(
        "No “tests not run” without operator gate evidence.",
        "Report test results if run; otherwise request Operator to run gates."
    )
    
    path.write_text(content, encoding="utf-8")

print("Updated .ai/AGENTS.md to allow Codex test execution.")
