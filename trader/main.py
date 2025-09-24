# trader/main.py
# Shim so workflows have a stable entrypoint.
# It looks for a likely crypto engine and runs it as __main__.

import os, sys, runpy, glob

HERE = os.path.dirname(__file__)

CANDIDATES = [
    "engine.py",
    "crypto_engine.py",
    "crypto_live.py",
    "crypto_main.py",
    "live.py",
    "bot.py",
]

# 1) Try known names in order
for name in CANDIDATES:
    path = os.path.join(HERE, name)
    if os.path.isfile(path):
        sys.argv = [path]
        runpy.run_path(path, run_name="__main__")
        raise SystemExit(0)

# 2) Fallback: any file that looks like crypto*
for path in sorted(glob.glob(os.path.join(HERE, "crypto*.py"))):
    if os.path.isfile(path):
        sys.argv = [path]
        runpy.run_path(path, run_name="__main__")
        raise SystemExit(0)

raise SystemExit("No crypto entrypoint found (looked for engine.py/crypto_*.py in trader/).")
