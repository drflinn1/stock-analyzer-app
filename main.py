#!/usr/bin/env python3
"""
Main unified runner for Crypto — Hourly 1-Coin Rotation.

Goals for this revision:
- Fix "ModuleNotFoundError: No module named 'requests'" by keeping imports simple
  and ensuring workflow installs requests explicitly (see YAML).
- Always create .state/ files so the artifact step never warns.
- Read Kraken keys from either (KRAKEN_API_KEY/KRAKEN_API_SECRET) or (KRAKEN_KEY/KRAKEN_SECRET).
- Be tolerant if your deeper trading code lives in a local package (PYTHONPATH set in YAML).
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

# ---------- Small helpers ----------
def env_str(name: str, default: str = "") -> str:
    val = os.getenv(name)
    return default if val is None else str(val)

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

def write_summary(data: Dict[str, Any]) -> None:
    # JSON
    SUMMARY_JSON.write_text(json.dumps(data, indent=2))
    # MD
    lines = [
        f"# Crypto — Hourly 1-Coin Rotation",
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
    k = os.getenv("KRAKEN_API_KEY") or os.getenv("KRAKEN_KEY") or ""
    s = os.getenv("KRAKEN_API_SECRET") or os.getenv("KRAKEN_SECRET") or ""
    return bool(k and s)

# ---------- Main ----------
def main() -> int:
    # Read env (all strings, keep simple)
    dry_run = env_str("DRY_RUN", "ON").upper()         # "ON" (paper) or "OFF" (live)
    buy_usd = env_str("BUY_USD", "25")
    tp_pct = env_str("TP_PCT", "5")
    stop_pct = env_str("STOP_PCT", "1")
    window_min = env_str("WINDOW_MIN", "60")
    slow_gain_req = env_str("SLOW_GAIN_REQ", "3")
    universe_pick = env_str("UNIVERSE_PICK", "")

    # Minimal preflight about keys (live only)
    live_requires_keys_ok = True
    note = ""
    if dry_run == "OFF":
        if not keys_present():
            live_requires_keys_ok = False
            note = (
                "LIVE requested but API keys are missing. "
                "Add KRAKEN_API_KEY/KRAKEN_API_SECRET (or KRAKEN_KEY/KRAKEN_SECRET) in Secrets."
            )

    # ---- INSERT your deeper trading logic here if you have a local package ----
    # If you have `from trader.crypto_engine import run_hourly_rotation`, it should
    # work now because the workflow sets PYTHONPATH to the repo root.
    #
    # Example skeleton (safe no-op in DRY_RUN or if keys missing):
    status = "ok"
    try:
        if dry_run == "OFF" and not live_requires_keys_ok:
            status = "skipped"
        else:
            # Replace this block with your real rotation logic call.
            # For now, simulate a quick run and ALWAYS write state files.
            time.sleep(0.5)
            status = "ok"
    except Exception as e:
        status = "error"
        note = f"Runtime error: {type(e).__name__}: {e}"

    # Always write .state artifacts so upload step never warns
    summary = {
        "when": now_iso(),
        "dry_run": dry_run,
        "buy_usd": buy_usd,
        "tp_pct": tp_pct,
        "stop_pct": stop_pct,
        "window_min": window_min,
        "slow_gain_req": slow_gain_req,
        "universe_pick": universe_pick or None,
        "status": status if live_requires_keys_ok else "skipped",
        "note": note,
    }
    write_summary(summary)

    if status == "error" or (dry_run == "OFF" and not live_requires_keys_ok):
        return 1
    LAST_OK.write_text(now_iso())
    return 0


if __name__ == "__main__":
    sys.exit(main())
