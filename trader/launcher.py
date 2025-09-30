# trader/launcher.py
# Auto-selects engine file and runs it. Order:
#   1) trader/crypto_engine.py
#   2) trader/main.py
#   3) trader/engine.py
# Prints a clear [go] line with the chosen file. If none exist, prints an error
# and lists the contents of trader/.

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

def main() -> int:
    chosen = pick_engine()
    if not chosen:
        print("ERROR: No engine file found. Looked for:", ", ".join(ENGINE_ORDER))
        list_trader_dir()
        return 1

    print(f"[launcher] Selected engine: {chosen}")
    print(f"[go] python {chosen}")

    # Pass through current environment (DRY_RUN, EXCHANGE_ID, etc.)
    try:
        # Use the same Python that launched this script
        cmd = [sys.executable, "-u", chosen]
        proc = subprocess.run(cmd, check=False)
        return proc.returncode
    except Exception as e:
        print(f"[launcher] Failed to run {chosen}: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
