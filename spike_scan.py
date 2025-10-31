#!/usr/bin/env python3
"""
Shim launcher so legacy workflows that call `python spike_scan.py`
will run the real scanner at tools/momentum_spike.py.

Also sanitizes numeric env vars so values like "25000# note" or "25,000"
won't crash the script.
"""

import os
import sys
import runpy

def _sanitize_env_number(name: str, default: str) -> None:
    raw = os.getenv(name, default)
    raw = raw.split("#", 1)[0].replace(",", "").strip()
    os.environ[name] = raw

# Sanitize common knobs (add more if you use them)
_sanitize_env_number("MIN_BASE_VOL_USD", "25000")
_sanitize_env_number("MIN_24H_PCT", "25")
_sanitize_env_number("MOMENTUM_RSI_MIN", "55")
_sanitize_env_number("MOMENTUM_RSI_MAX", "80")
_sanitize_env_number("MOMENTUM_EMA_WINDOW", "20")
_sanitize_env_number("MAX_CANDIDATES", "10")

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_PATH = os.path.join(REPO_DIR, "tools")
TARGET = os.path.join(TOOLS_PATH, "momentum_spike.py")

if not os.path.exists(TARGET):
    sys.stderr.write(f"[shim] Not found: {TARGET}\n")
    sys.exit(2)

sys.path.insert(0, TOOLS_PATH)
runpy.run_path(TARGET, run_name="__main__")
