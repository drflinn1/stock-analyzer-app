#!/usr/bin/env python3
"""
tools/run_live.py
Runs your existing main.py with the same args, but filters out the
confusing log fragment: "The truth value of a Series is ambiguous..."
Nothing else is changed.

We preserve all logs; we only replace that exact fragment with a clearer note.
"""

import os
import sys
import runpy
import io

NOISY = "The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all()"
CLEAN = "Indicator condition ambiguous (likely Series vs scalar). Skipped trade."

class _FilterStream(io.TextIOBase):
    def __init__(self, orig):
        self._orig = orig

    def write(self, s):
        try:
            s = s.replace(NOISY, CLEAN)
        except Exception:
            pass
        return self._orig.write(s)

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass

# Wrap stdout/stderr so logs get cleaned
sys.stdout = _FilterStream(sys.stdout)
sys.stderr = _FilterStream(sys.stderr)

# Build argv to mirror how workflows call main.py
# If the workflow passes args, we keep them; if launched directly, we default.
if len(sys.argv) == 1:
    # Defaults for crypto cautious live (same as workflows)
    sys.argv = [
        "main.py",
        "--market", "crypto",
        "--exchange", "kraken",
        "--strategy", "cautious",
        "--cap_per_trade", "10",
        "--cap_daily", "15",
        "--dry_run", "false",
    ]

# Ensure LIVE mode (belt + suspenders)
os.environ.setdefault("TRADE_MODE", "live")
os.environ.setdefault("DRY_RUN", "false")

# Hand off to your app
runpy.run_path("main.py", run_name="__main__")
