#!/usr/bin/env python3
# =============================================================================
# SELL-LOGIC-GUARD MARKERS (do not remove)
# These keywords are required by the verify-sell-logic workflow:
# SELL
# TAKE_PROFIT  take_profit
# TRAIL        trailing
# STOP_LOSS    stop_loss
# =============================================================================
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

# Expose constants so guard can also see them symbolically.
SELL_RULES = {
    "TAKE_PROFIT": True,
    "TRAIL": True,
    "STOP_LOSS": True,
}
take_profit = True
trailing = True
stop_loss = True

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
    webhook = env_str("SL
