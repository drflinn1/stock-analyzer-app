#!/usr/bin/env python3
"""
Summarize current positions from .state/positions.json.
- Prints totals and a per-symbol table (qty, avg cost, last price, unrealized P/L).
- Writes CSV to data/positions_summary.csv
- Uses public CCXT price fetch (no keys needed).
"""
import os, json, csv, math
from typing import Dict, Any, List

STATE_DIR = ".state"
POS_FILE = os.path.join(STATE_DIR, "positions.json")
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "positions_summary.csv")

EXCHANGE_NAME = os.getenv("EXCHANGE", "kraken").lower()
QUOTE = os.getenv("QUOTE", "USD").upper()

def load_positions() -> Dict[str, List[Dict[str, Any]]]:
    if not os.path.exists(POS_FILE):
        print(f"[summary] No positions file at {POS_FILE}")
        return {}
    try:
        with open(POS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[summary] Failed to read {POS_FILE}: {e}")
        return {}

def mk_exchange():
    try:
        import ccxt  # type: ignore
    except Exception as e:
        print("[summary] ccxt not installed; ensure requirements.txt includes ccxt")
        raise
    if EXCHANGE_NAME != "kraken":
        raise RuntimeError(f"Only kraken supported in this helper; got {EXCHANGE_NAME}")
    ex = ccxt.kraken({"enableRateLimit": True})
    ex.load_markets()
    return ex

def last_price(ex, symbol: str) -> float:
    try:
        t = ex.fetch_ticker(symbol)
        return float(t.get("last") or t.get("close") or 0.0)
    except Exception:
        return 0.0

def main():
    pos = load_positions()
    total_symbols = 0
    total_lots = 0
    total_notional = 0.0

    if not pos:
        print("[summary] No positions found.")
        return

    ex = mk_exchange()

    rows = []
    header = [
        "symbol","lots","total_qty","avg_cost_usd","last_px_usd",
        "notional_usd","unreal_pnl_usd","unreal_pnl_pct"
    ]

    for sym, lots in pos.items():
        if not isinstance(lots, list) or not lots:
            continue
        valid = [l for l in lots if isinstance(l, dict) and "qty" in l and "cost" in l]
        if not valid:
            continue
        total_symbols += 1
        total_lots += len(valid)
        qty = sum(float(l["qty"]) for l in valid)
        if qty <= 0:
            continue
        wcost = sum(float(l["qty"]) * float(l["cost"]) for l in valid) / qty
        px = last_price(ex, sym)
        notion = qty * px
        pnl = qty * (px - wcost)
        pnl_pct = ((px - wcost) / wcost * 100.0) if wcost > 0 else 0.0
        total_notional += notion

        rows.append([
            sym, len(valid), f"{qty:.8f}", f"{wcost:.8f}", f"{px:.8f}",
            f"{notion:.2f}", f"{pnl:.2f}", f"{pnl_pct:.2f}"
        ])

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"[summary] symbols={total_symbols} lots={total_lots} notional≈${total_notional:.2f}")
    print(f"[summary] wrote {OUT_CSV}")
    print("[summary] Top 10 lines:")
    for r in rows[:10]:
        print("  ", r)

if __name__ == "__main__":
    main()
