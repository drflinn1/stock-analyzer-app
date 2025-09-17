#!/usr/bin/env python3
import os, json
from typing import Dict, Any, List

STATE_DIR = ".state"
POS_FILE = os.path.join(STATE_DIR, "positions.json")
EXCHANGE = os.getenv("EXCHANGE", "kraken").lower()
QUOTE = os.getenv("QUOTE", "USD").upper()

TAKE_PROFIT_PCT       = float(os.getenv("TAKE_PROFIT_PCT", "2.0"))
TRAILING_ACTIVATE_PCT = float(os.getenv("TRAILING_ACTIVATE_PCT", "1.0"))
TRAILING_DELTA_PCT    = float(os.getenv("TRAILING_DELTA_PCT", "1.0"))

def load_positions() -> Dict[str, List[Dict[str, Any]]]:
    if not os.path.exists(POS_FILE): return {}
    with open(POS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data if isinstance(data, dict) else {}

def mk_exchange():
    import ccxt  # type: ignore
    if EXCHANGE != "kraken":
        raise RuntimeError(f"Only kraken supported here; got {EXCHANGE}")
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
    if not pos:
        print("No positions found.")
        return
    ex = mk_exchange()
    print(f"Rules → TP={TAKE_PROFIT_PCT:.2f}%  TRAIL_ACT={TRAILING_ACTIVATE_PCT:.2f}%  PULLBACK={TRAILING_DELTA_PCT:.2f}%")
    print("symbol  chg%  TP_hit  trail_ready  would_sell_if_pulled_back")
    any_rows = False
    for sym, lots in pos.items():
        lots = [l for l in lots if isinstance(l, dict) and "qty" in l and "cost" in l]
        if not lots: continue
        qty = sum(float(l["qty"]) for l in lots)
        if qty <= 0: continue
        wcost = sum(float(l["qty"])*float(l["cost"]) for l in lots)/qty
        px = last_price(ex, sym)
        if px <= 0 or wcost <= 0: continue
        chg_pct = (px - wcost)/wcost*100.0
        tp_hit = chg_pct >= TAKE_PROFIT_PCT
        trail_ready = chg_pct >= TRAILING_ACTIVATE_PCT
        would_sell_on_pullback = trail_ready
        print(f"{sym:10s} {chg_pct:6.2f}%  {str(tp_hit):5s}  {str(trail_ready):11s}  {str(would_sell_on_pullback):23s}")
        any_rows = True
    if not any_rows:
        print("No symbols qualified for evaluation.")

if __name__ == "__main__":
    main()
