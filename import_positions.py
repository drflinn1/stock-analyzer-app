#!/usr/bin/env python3
import os, json, datetime
from typing import Dict, Any, List

STATE_DIR = ".state"
POS_FILE = os.path.join(STATE_DIR, "positions.json")
QUOTE = os.getenv("QUOTE", "USD").upper()
EXCHANGE_NAME = os.getenv("EXCHANGE", "kraken").lower()

def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

try:
    import ccxt  # type: ignore
except Exception as e:
    print("ccxt not available; ensure requirements.txt installs it")
    raise

def mk_exchange():
    if EXCHANGE_NAME != "kraken":
        raise RuntimeError(f"Only kraken supported here; got {EXCHANGE_NAME}")
    ex = ccxt.kraken({
        "apiKey": os.getenv("KRAKEN_API_KEY", ""),
        "secret": os.getenv("KRAKEN_API_SECRET", ""),
        "enableRateLimit": True,
    })
    ex.load_markets()
    return ex

def main():
    ex = mk_exchange()
    bal = ex.fetch_balance()  # dict with 'total' by currency
    totals = bal.get("total") or {}
    markets = ex.markets

    positions: Dict[str, List[Dict[str, Any]]] = {}

    def price(sym: str) -> float:
        try:
            t = ex.fetch_ticker(sym)
            p = t.get("last") or t.get("close") or 0.0
            return float(p or 0.0)
        except Exception:
            return 0.0

    for base_code, amt in totals.items():
        try:
            qty = float(amt or 0.0)
        except Exception:
            qty = 0.0
        if qty <= 0: 
            continue
        if base_code in (QUOTE, "USDT", "USD", "ZUSD", "USDC"):
            continue

        sym = f"{base_code}/{QUOTE}"
        if sym not in markets or not markets[sym].get("active", True):
            # skip assets without a direct QUOTE pair
            continue

        px = price(sym)
        if px <= 0:
            continue

        entry = datetime.datetime.utcnow().isoformat()
        positions[sym] = positions.get(sym, [])
        positions[sym].append({"qty": qty, "cost": px, "entry": entry})
        print(f"Imported {sym}: qty={qty:.8f} cost≈{px:.8f}")

    _save_json(POS_FILE, positions)
    print(f"Wrote {POS_FILE} with {sum(len(v) for v in positions.values())} lots "
          f"across {len(positions)} symbols.")

if __name__ == "__main__":
    main()
