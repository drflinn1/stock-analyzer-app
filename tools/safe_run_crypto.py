#!/usr/bin/env python3
"""
Strict runner for crypto live: if any pandas Series is used in a boolean
context (e.g., `if series:` or `series > x` without reduction), raise and
print a full traceback so we can see the exact file/line to fix.

This does not modify your source files; it only wraps the run.
"""

import os
import sys
import traceback
import warnings

# Make warnings loud
warnings.filterwarnings("error")

# Hard fail on pandas ambiguous truth evaluation
try:
    import pandas as pd  # type: ignore
    from pandas.core.series import Series as _PdSeries  # type: ignore

    def _strict_bool(self):  # type: ignore[no-redef]
        """
        Any attempt to evaluate a Series in boolean context will raise with
        a clear traceback so we know where to fix in the codebase.
        """
        tb = "".join(traceback.format_stack(limit=12))
        raise ValueError(
            "STRICT MODE: A pandas Series was used in a boolean context.\n"
            "This usually means code like `if series:` or comparing a Series "
            "without reducing via .any()/.all()/.iloc[-1].\n\nTraceback:\n" + tb
        )

    # Monkey-patch pandas Series.__bool__ to force an error + traceback
    _PdSeries.__bool__ = _strict_bool  # type: ignore[attr-defined]

except Exception as e:  # pragma: no cover
    print("Warning: could not enable strict pandas mode:", e, file=sys.stderr)

# Build argv for your app exactly as in the workflow
sys.argv = [
    "main.py",
    "--market", "crypto",
    "--exchange", "kraken",
    "--strategy", "cautious",
    "--cap_per_trade", "10",
    "--cap_daily", "15",
    "--dry_run", "false",
]

# Ensure live mode via env (belt + suspenders)
os.environ.setdefault("TRADE_MODE", "live")
os.environ.setdefault("DRY_RUN", "false")
os.environ.setdefault("MARKET", "crypto")
os.environ.setdefault("EQUITY_BROKER", "none")
os.environ.setdefault("SYMBOLS", "BTC-USD")
os.environ.setdefault("EXCHANGE", "kraken")
os.environ.setdefault("CCXT_EXCHANGE", "kraken")

# Run your existing script under strict mode
import runpy
runpy.run_path("main.py", run_name="__main__")
