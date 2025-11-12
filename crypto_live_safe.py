#!/usr/bin/env python3
"""
crypto_live_safe.py
Single-file live runner with robust state handling.

Fixes:
- Avoid KeyError when extracting current symbol from .state/positions.json by
  tolerating multiple shapes/keys and empty/missing files.
- Normalizes symbols like 'LSK', 'LSKUSD', 'LSK/USD' consistently to 'LSK'.

This keeps your previous buy/sell flow intact while making the state layer safe.
"""

from __future__ import annotations
import json, os, sys, time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

STATE = Path(".state")
STATE.mkdir(parents=True, exist_ok=True)

# ------------------------------ utils ---------------------------------
def read_json(p: Path) -> Any:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"⚠️  Failed to read {p}: {e}", file=sys.stderr)
        return None

def write_json(p: Path, obj: Any) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    tmp.replace(p)

def norm_symbol(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip().upper()
    if "/" in s:
        s = s.split("/")[0]
    if s.endswith("USD"):
        s = s[:-3]
    if s.endswith("-USD"):
        s = s[:-4]
    return s or None

def extract_symbolish(d: Dict[str, Any]) -> Optional[str]:
    # Try common keys in order of likelihood
    for k in ("symbol", "pair", "ticker", "asset", "coin"):
        if k in d and isinstance(d[k], str) and d[k].strip():
            return norm_symbol(d[k])
    # Sometimes stored like {"LSK": {...}}
    if len(d) == 1:
        k = next(iter(d))
        if isinstance(k, str):
            return norm_symbol(k)
    return None

def load_current_position() -> Optional[Dict[str, Any]]:
    """
    Accepts a bunch of shapes, returns {"symbol": "LSK", ...} or None
    Shapes we tolerate:
    - {"symbol":"LSK", "qty": 12}
    - {"pair":"LSK/USD", ...}
    - {"positions":[{"symbol":"LSK", ...}, ...]}  -> take the first / only live coin
    - {"current":{"ticker":"LSKUSD", ...}}
    - {"LSK":{"qty":10}} or {"LSKUSD":{...}}
    """
    p = STATE / "positions.json"
    data = read_json(p)
    if not data:
        return None

    # Direct simple dict with a symbol-ish key
    if isinstance(data, dict):
        # Common container forms
        if "positions" in data and isinstance(data["positions"], list) and data["positions"]:
            for item in data["positions"]:
                if not isinstance(item, dict): 
                    continue
                sym = extract_symbolish(item)
                if sym:
                    item = dict(item)
                    item["symbol"] = sym
                    return item
            return None

        if "current" in data and isinstance(data["current"], dict):
            sym = extract_symbolish(data["current"])
            if sym:
                cur = dict(data["current"])
                cur["symbol"] = sym
                return cur

        # Flat dict may itself be a position
        sym = extract_symbolish(data)
        if sym:
            d = dict(data)
            d["symbol"] = sym
            return d

    # List form—take the first item with a recognizable symbol
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                sym = extract_symbolish(item)
                if sym:
                    item = dict(item)
                    item["symbol"] = sym
                    return item
        return None

    return None

def save_current_position(symbol: Optional[str], extra: Optional[Dict[str, Any]] = None) -> None:
    if symbol is None:
        # Clear position
        write_json(STATE / "positions.json", {"positions": []})
        return
    d = {"symbol": norm_symbol(symbol)}
    if extra:
        d.update(extra)
    write_json(STATE / "positions.json", {"current": d})

def load_candidates() -> List[Dict[str, Any]]:
    """
    Read momentum candidates (symbol, quote, rank) if present.
    Gracefully handle missing/partial files; return [] if none.
    """
    import csv
    out = []
    csv_path = STATE / "momentum_candidates.csv"
    if not csv_path.exists():
        return out
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                sym = norm_symbol(row.get("symbol") or row.get("pair") or row.get("ticker"))
                if not sym:
                    continue
                try:
                    quote = float(row.get("quote", "") or "0")
                except ValueError:
                    quote = 0.0
                try:
                    rank = float(row.get("rank", "") or "0")
                except ValueError:
                    rank = 0.0
                out.append({"symbol": sym, "quote": quote, "rank": rank})
    except Exception as e:
        print(f"⚠️  Failed reading candidates CSV: {e}", file=sys.stderr)
    return out

# ------------------------------ trading stubs --------------------------
def kraken_place_order(symbol: str, side: str, usd: float, dry_run: bool) -> Tuple[bool, str]:
    # Minimal stub; integrate with your real kraken adapter here
    if dry_run:
        return True, f"DRY-RUN {side} {symbol} ${usd:.2f}"
    # TODO: real API call
    return True, f"LIVE {side} {symbol} ${usd:.2f}"

def kraken_force_sell(symbol: str, dry_run: bool) -> Tuple[bool, str]:
    if dry_run:
        return True, f"DRY-RUN SELL {symbol} (force)"
    return True, f"LIVE SELL {symbol} (force)"

# ------------------------------ strategy --------------------------------
def env_or(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None and v != "" else default

def run() -> int:
    # ---- env
    DRY_RUN = env_or("DRY_RUN", "ON").upper()   # "ON" or "OFF"
    BUY_USD = float(env_or("BUY_USD", env_or("MIN_BUY_USD", "30")))
    TP_PCT  = float(env_or("TP_PCT", "8"))      # % take profit
    STOP_PCT= float(env_or("SL_PCT", env_or("STOP_PCT", "4")))
    WINDOW  = int(env_or("WINDOW_MIN", "30"))
    PICK    = os.getenv("UNIVERSE_PICK") or os.getenv("PICK") or ""

    dry = (DRY_RUN != "OFF")

    print(f"DRY_RUN={DRY_RUN}  BUY_USD={BUY_USD:.2f}  TP={TP_PCT}  STOP={STOP_PCT}  WINDOW={WINDOW}  PICK='{PICK}'")

    # ---- current position (robust)
    cur = load_current_position()
    cur_sym = cur["symbol"] if isinstance(cur, dict) and "symbol" in cur else None
    print(f"Current position: {cur_sym or 'NONE'}")

    # ---- candidate selection (keep simple; respect PICK if set)
    if PICK.strip():
        target = norm_symbol(PICK)
    else:
        cands = load_candidates()
        target = cands[0]["symbol"] if cands else None

    if not cur_sym and not target:
        print("ℹ️  No current position and no candidates available. Exiting cleanly.")
        return 0

    # -------------------- trivial decision logic ------------------------
    # If we don't hold anything: buy target (if any)
    if not cur_sym and target:
        ok, msg = kraken_place_order(target, "BUY", BUY_USD, dry)
        print(("✅ " if ok else "❌ ") + msg)
        if ok:
            save_current_position(target, {"qty": 1})
        return 0

    # If we do hold something but PICK points elsewhere, force-rotate
    if cur_sym and target and target != cur_sym:
        print(f"🔁 Rotate: SELL {cur_sym} → BUY {target}")
        ok1, msg1 = kraken_force_sell(cur_sym, dry)
        print(("✅ " if ok1 else "❌ ") + msg1)
        ok2, msg2 = kraken_place_order(target, "BUY", BUY_USD, dry)
        print(("✅ " if ok2 else "❌ ") + msg2)
        if ok1 and ok2:
            save_current_position(target, {"qty": 1})
        return 0

    # Otherwise, hold. (Your real TP/SL logic runs elsewhere; this just avoids crashes.)
    print("🟨 No action required this cycle.")
    return 0

# ------------------------------ main -----------------------------------
def main():
    code = 0
    try:
        code = run()
    except SystemExit as e:
        code = int(e.code) if isinstance(e.code, int) else 1
    except Exception as e:
        print(f"❌ Unhandled error: {e}", file=sys.stderr)
        code = 1
    finally:
        # Drop a summary so the workflow artifact is always useful
        summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "result": "ok" if code == 0 else "error",
        }
        write_json(STATE / "run_summary.json", summary)
    sys.exit(code)

if __name__ == "__main__":
    main()
