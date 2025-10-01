# trader/launcher.py
# Auto-select engine and run it.
# Order:
#   1) trader/crypto_engine.py
#   2) trader/main.py
#   3) trader/engine.py
#
# If trader/__init__.py exists AND the chosen engine is trader/crypto_engine.py,
# run it as a module:  python -m trader.crypto_engine
# Otherwise run the file directly.

from __future__ import annotations
import os, sys, subprocess

ENGINE_ORDER = [
    "trader/crypto_engine.py",
    "trader/main.py",
    "trader/engine.py",
]

def pick_engine() -> str | None:
    for path in ENGINE_ORDER:
        if os.path.isfile(path):
            return path
    return None

def list_trader_dir() -> None:
    print("\n--- trader/ directory listing ---")
    if not os.path.isdir("trader"):
        print("(missing 'trader' directory entirely)")
        return
    for root, dirs, files in os.walk("trader"):
        for f in sorted(files):
            print(os.path.join(root, f))

def run_python(target: str) -> int:
    has_pkg = os.path.isfile("trader/__init__.py")
    # Prefer module execution for crypto_engine when we have a package
    if has_pkg and target == "trader/crypto_engine.py":
        print("[launcher] Package detected; running as module")
        print("[go] python -m trader.crypto_engine")
        cmd = [sys.executable, "-u", "-m", "trader.crypto_engine"]
    else:
        print(f"[go] python {target}")
        cmd = [sys.executable, "-u", target]

    proc = subprocess.run(cmd, check=False)
    return proc.returncode

def main() -> int:
    chosen = pick_engine()
    if not chosen:
        print("ERROR: No engine file found. Looked for:", ", ".join(ENGINE_ORDER))
        list_trader_dir()
        return 1

    print(f"[launcher] Selected engine: {chosen}")
    try:
        return run_python(chosen)
    except Exception as e:
        print(f"[launcher] Failed to run {chosen}: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
