#!/usr/bin/env python3
"""
Main unified runner for Crypto — Hourly 1-Coin Rotation.

✅ Always writes .state/run_summary.json + .state/run_summary.md
✅ Compatible with Kraken key names: KRAKEN_API_KEY / KRAKEN_API_SECRET or KRAKEN_KEY / KRAKEN_SECRET
✅ Safe default: DRY_RUN=ON (paper)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# ---------- Paths ----------
STATE_DIR = Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_JSON = STATE_DIR / "run_summary.json"
SUMMARY_MD = STATE_DIR / "run_summary.md"
LAST_OK = STATE_DIR / "last_ok.txt"

# ---------- Helpers ----------
def env_str(name: str, default: str = "") -> str:
    """Read environment variable as string with default."""
    val = os.getenv(name)
    return default if val is None else str(val)

def now_iso() -> str:
    """Return current UTC timestamp as string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

def write_summary(data: Dict[str, Any]) -> None:
    """Write JSON + Markdown summaries to .state."""
    SUMMARY_JSON.write_text(json.dumps(data, indent=2))
    lines = [
        "# Crypto — Hourly 1-Coin Rotation",
        f"**When:** {data.get('when')}",
        f"**DRY_RUN:** {data.get('dry_run')}",
        f"**BUY_USD:** {data.get('buy_usd')}",
        f"**TP_PCT:** {data.get('tp_pct')}%",
        f"**STOP_PCT:** {data.get('stop_pct')}%",
        f"**WINDOW_MIN:** {data.get('window_min')} min",
        f"**SLOW_GAIN_REQ:** {data.get('slow_gain_req')}%",
        f"**UNIVERSE_PICK:** {data.get('universe_pick') or '<auto>'}",
        "",
        f"**Status:** {data.get('status')}",
        f"**Note:** {data.get('note') or '-'}",
    ]
    SUMMARY_MD.write_text("\n".join(lines))

def keys_present() -> bool:
    """Return True if Kraken API keys exist."""
    k = os.getenv("KRAKEN_API_KEY") or os.getenv("KRAKEN_KEY") or ""
    s = os.getenv("KRAKEN_API_SECRET") or os.getenv("KRAKEN_SECRET") or ""
    return bool(k and s)

# ---------- Main ----------
def main() -> int:
    dry_run = env_str("DRY_RUN", "ON").upper()
    buy_usd = env_str("BUY_USD", "25")
    tp_pct = env_str("TP_PCT", "5")
    stop_pct = env_str("STOP_PCT", "1")
    window_min = env_str("WINDOW_MIN", "60")
    slow_gain_req = env_str("SLOW_GAIN_REQ", "3")
    universe_pick = env_str("UNIVERSE_PICK", "")

    note = ""
    live_keys_ok = keys_present()

    # Safety check for live mode
    if dry_run == "OFF" and not live_keys_ok:
        note = "LIVE requested but API keys missing — aborting orders."
        status = "skipped"
    else:
        # Placeholder for actual trading logic
        status = "ok"
        try:
            # --- Example simulation ---
            time.sleep(0.5)
            # Insert your trading logic call here later
        except Exception as e:
            status = "error"
            note = f"{type(e).__name__}: {e}"

    # Always create .state artifacts
    summary = {
        "when": now_iso(),
        "dry_run": dry_run,
        "buy_usd": buy_usd,
        "tp_pct": tp_pct,
        "stop_pct": stop_pct,
        "window_min": window_min,
        "slow_gain_req": slow_gain_req,
        "universe_pick": universe_pick or None,
        "status": status,
        "note": note,
    }

    write_summary(summary)

    if status == "error" or (dry_run == "OFF" and not live_keys_ok):
        return 1

    LAST_OK.write_text(now_iso())
    return 0


if __name__ == "__main__":
    sys.exit(main())
