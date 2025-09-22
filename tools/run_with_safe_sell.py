# tools/run_with_safe_sell.py
"""
Wrap the live run to reduce 'insufficient funds' buy errors by reserving a tiny cushion,
and by cancelling any stale open orders before attempting sells.
"""

import os
import subprocess
import sys

# read/derive knobs (all provided from workflow env)
eps_pct = float(os.getenv("SELL_SAFETY_EPS_PCT", "0.25"))
cancel_before_sell = os.getenv("CANCEL_OPEN_ORDERS_BEFORE_SELL", "true").lower() == "true"
fee_cushion = float(os.getenv("FEE_BUY_BUFFER_PCT", "0.50"))

env = os.environ.copy()
env["SELL_SAFETY_EPS_PCT"] = f"{eps_pct:.2f}"
env["FEE_BUY_BUFFER_PCT"] = f"{fee_cushion:.2f}"
env["CANCEL_OPEN_ORDERS_BEFORE_SELL"] = "true" if cancel_before_sell else "false"

cmd = [sys.executable, "-u", "main.py"]
print(f"[wrapper] SELL_SAFETY_EPS_PCT={env['SELL_SAFETY_EPS_PCT']} FEE_BUY_BUFFER_PCT={env['FEE_BUY_BUFFER_PCT']} cancel_before_sell={cancel_before_sell}")

proc = subprocess.run(cmd, env=env)
sys.exit(proc.returncode)
