# trader/engine.py
# Minimal safe shim so the Crypto Live workflow has a valid entrypoint.
# It logs and exits cleanly (code 0). Replace later with your real crypto engine.

import os, sys, time

DRY_RUN = os.environ.get("DRY_RUN", "false").lower() in ("1","true","yes","y","on")

print("=== Crypto Engine Shim ===")
print("No active crypto engine detected. This shim exits cleanly so workflows don't fail.")
print(f"DRY_RUN={DRY_RUN}")
print("Next step: replace trader/engine.py with your real crypto engine "
      "(or point the workflow to the correct file).")
sys.exit(0)
