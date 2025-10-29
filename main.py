#!/usr/bin/env python3
"""
Unified runner wrapper for the Crypto Live workflows.

Goals:
- Respect env toggles (DRY_RUN, RUN_SWITCH).
- Call your existing engine (default: trader/crypto_engine.py) if it exists.
- Always create helpful artifacts under .state/ for debugging:
    .state/run_summary.json
    .state/run_summary.md
    .state/last_ok.txt (when logic executes)
- Optional Slack notification of the summary.
- Never crash the workflow on simple setup issues (gives a clear summary instead).

This file is intentionally self-contained and safe to drop in at repo root.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any

# ---------- helpers ----------

STATE_DIR = Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_JSON = STATE_DIR / "run_summary.json"
SUMMARY_MD = STATE_DIR / "run_summary.md"
LAST_OK = STATE_DIR / "last_ok.txt"

def env_str(name: str, default: str = "") -> str:
    val = os.getenv(name, default)
    if isinstance(val, bytes):
        val = val.decode("utf-8", "ignore")
    return str(val)

def write_summary(data: Dict[str, Any]) -> None:
    SUMMARY_JSON.write_text(json.dumps(data, indent=2))
    # Minimal MD for GitHub Job Summary
    lines = [
        f"**When:** {data.get('when')}",
        f"**DRY_RUN:** {data.get('DRY_RUN')}",
        f"**RUN_SWITCH:** {data.get('RUN_SWITCH')}",
        f"**Entry:** {data.get('entrypoint')}",
        f"**Engine Executed:** {data.get('engine_executed')}",
        f"**Notes:** {data.get('notes','')}",
    ]
    SUMMARY_MD.write_text("\n\n".join(lines) + "\n")

def post_slack(text: str) -> None:
    import requests  # lazy import in case not installed
    webhook = env_str("SLACK_WEBHOOK_URL")
    if not webhook:
        return
    try:
        requests.post(webhook, json={"text": text}, timeout=10)
    except Exception:
        # Do not fail the job because Slack hiccuped
        pass

def file_exists(path: str) -> bool:
    return Path(path).is_file()

def run_engine_by_path(path: str, dry_run: str) -> int:
    """
    Execute the engine python file as a separate process.
    Returns the process return code.
    """
    cmd = [sys.executable, "-u", path]
    env = os.environ.copy()
    env["DRY_RUN"] = dry_run  # ensure toggle is visible to engine
    print(f"[runner] exec: {' '.join(cmd)}")
    try:
        return subprocess.run(cmd, env=env, check=False).returncode
    except Exception as e:
        print(f"[runner] engine error: {e}", file=sys.stderr)
        return 1

# ---------- main flow ----------

def main() -> int:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    dry_run = env_str("DRY_RUN", "ON").upper()
    run_switch = env_str("RUN_SWITCH", "ON").upper()
    entrypoint = env_str("ENTRYPOINT", "trader/crypto_engine.py")

    summary: Dict[str, Any] = {
        "when": now,
        "DRY_RUN": dry_run,
        "RUN_SWITCH": run_switch,
        "entrypoint": entrypoint,
        "engine_executed": False,
        "notes": "",
    }

    # Always ensure artifacts folder exists
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # Soft gate to avoid surprises
    if run_switch not in ("ON", "OFF"):
        run_switch = "ON"  # default safe
        summary["notes"] += "RUN_SWITCH invalid; defaulted to ON. "

    if run_switch == "OFF":
        summary["notes"] += "RUN_SWITCH=OFF → skipping trading logic.\n"
        write_summary(summary)
        post_slack(f"🔕 CryptoBot skipped (RUN_SWITCH=OFF, DRY_RUN={dry_run}).")
        print("[runner] Skipped by RUN_SWITCH=OFF")
        return 0

    # If the requested entrypoint exists, run it. Otherwise, try a few fallbacks.
    candidates = [entrypoint, "trader/main.py", "bot/main.py", "engine.py"]
    engine_rc = 0
    executed = False
    for candidate in candidates:
        if file_exists(candidate):
            engine_rc = run_engine_by_path(candidate, dry_run)
            executed = True
            break

    summary["engine_executed"] = executed

    if not executed:
        # No engine found → graceful no-op with clear diagnostics
        msg = (
            "No engine file found. Looked for: " + ", ".join(candidates) +
            ". Set ENTRYPOINT (repo Variable) to the actual path of your bot's engine."
        )
        summary["notes"] += msg + "\n"
        print("[runner]", msg)

    # Persist artifacts / summary regardless
    write_summary(summary)

    # Mark the run as having executed logic at least once
    if executed and engine_rc == 0:
        LAST_OK.write_text(now + "\n")

    # Slack (short)
    status_emoji = "🟢" if engine_rc == 0 else "🔴"
    engine_bit = "yes" if executed else "no"
    post_slack(f"{status_emoji} CryptoBot run • engine:{engine_bit} • DRY_RUN:{dry_run} • RUN_SWITCH:{run_switch}")

    return engine_rc

if __name__ == "__main__":
    sys.exit(main())
