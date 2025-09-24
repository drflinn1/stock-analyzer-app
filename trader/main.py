# trader/main.py
# Forwarder shim: if a workflow calls trader/main.py, run crypto_engine.py instead.

import os, sys, runpy

HERE = os.path.dirname(__file__)
TARGET = os.path.join(HERE, "crypto_engine.py")

if not os.path.isfile(TARGET):
    raise SystemExit("crypto_engine.py not found next to main.py — please add trader/crypto_engine.py")

# Ensure __name__ == "__main__" behavior in the target
sys.argv = [TARGET]
runpy.run_path(TARGET, run_name="__main__")
