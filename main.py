#!/usr/bin/env python3
"""
Main wrapper for the crypto live workflows.

- Respects env toggles DRY_RUN and RUN_SWITCH.
- Calls your real engine (ENTRYPOINT env, default trader/crypto_engine.py).
- Writes artifacts into .state/ for quick debugging.
- Optional Slack notification if SLACK_WEBHOOK_URL is set.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any

STATE_DIR = Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_JSON = STATE_DIR / "run_summary.json"
SUMMARY_MD = STATE_DIR / "run_summary.md"
LAST_OK = STATE_DIR / "last_ok.txt"

def env_str(name: str, default: str = "") -> str:
    val = os.getenv(name, default)
    return "" if val is None else str(val)

def write_summary(data: Dict[str, Any]) -> None:
    SUMMARY_JSON.write_text(json.dumps(data, indent=2))
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
    webhook = env_str("SLACK_WEBHOOK_URL")
    if not webhook:
        return
    try:
        import requests
        requests.post(webhook, json={"text": text}, timeout=10)
    except Exception:
        pass

def file_exists(path: str) -> bool:
    return Path(path).is_file()

def run_engine(path: str, dry_run: str) -> int:
    cmd = [sys.executable, "-u", path]
    env = os.environ.copy()
    env["DRY_RUN"] = dry_run
    print(f"[runner] exec: {' '.join(cmd)}")
    try:
        return subprocess.run(cmd, env=env, check=False).returncode
    except Exception as e:
        print(f"[runner] engine error: {e}", file=sys.stderr)
        return 1

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

    if run_switch not in ("ON", "OFF"):
        run_switch = "ON"
        summary["notes"] += "RUN_SWITCH invalid; defaulted to ON. "

    if run_switch == "OFF":
        summary["notes"] += "RUN_SWITCH=OFF -> skipping trading logic.\n"
        write_summary(summary)
        post_slack(f"Muted: bot skipped (RUN_SWITCH=OFF, DRY_RUN={dry_run}).")
        print("[runner] Skipped by RUN_SWITCH=OFF")
        return 0

    candidates = [entrypoint, "trader/main.py", "bot/main.py", "engine.py"]
    executed = False
    rc = 0
    for c in candidates:
        if file_exists(c):
            rc = run_engine(c, dry_run)
            executed = True
            break

    summary["engine_executed"] = executed
    if not executed:
        summary["notes"] += (
            "No engine file found. Looked for: "
            + ", ".join(candidates)
            + ". Set ENTRYPOINT (repo Variable) to your engine path.\n"
        )
        print("[runner] No engine file found. Set ENTRYPOINT variable.")

    write_summary(summary)
    if executed and rc == 0:
        LAST_OK.write_text(now + "\n")

    status = "OK" if rc == 0 else "ERR"
    post_slack(f"Crypto run {status} • engine:{'yes' if executed else 'no'} • DRY_RUN:{dry_run} • RUN_SWITCH:{run_switch}")
    return rc

if __name__ == "__main__":
    sys.exit(main())
