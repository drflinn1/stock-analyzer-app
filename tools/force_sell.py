#!/usr/bin/env python3
"""
Force sell utility:
- Accepts SYMBOL = 'SOON', 'SOONUSD', 'SOON/USD', or 'ALL'
- Tries market first; if Kraken blocks with price protection, retries with a tight limit
"""

import os, sys, math
from trader.kraken import Kraken  # your adapter

def norm(s: str) -> str:
    s = (s or "").upper().strip().replace(" ", "")
    if s in ("", "ALL"): return s
    return s if "/" in s else (s.replace("USD","") + "/USD") if s.endswith("USD") else (s + "/USD")

def main():
    live = (os.getenv("DRY_RUN","ON").upper()=="OFF")
    sym_in = os.getenv("SYMBOL","ALL")
    limit_slip = float(os.getenv("LIMIT_SLIP_PCT","1.5"))/100.0
    K = Kraken(live=live)

    holdings = K.open_positions()  # { "XXX/USD": amount }
    if not holdings:
        print("No holdings found."); return

    targets = []
    if sym_in.upper() == "ALL":
        targets = [ (s,a) for s,a in holdings.items() if a>0 ]
    else:
        want = norm(sym_in)
        amt  = holdings.get(want) or holdings.get(want.replace("/","")) or 0
        if amt<=0:
            print(f"Symbol {want} not found or zero size; abort.")
            return
        targets = [(want, amt)]

    for sym, amt in targets:
        if amt<=0: continue
        print(f"[FORCE] Attempt market sell {amt} {sym} ...")
        ok, info = K.try_market_sell(sym, amt)
        if ok:
            print(f"Sold {sym} @ market.")
            continue
        # retry with tight limit through the book
        px = K.quote(sym).get("bid") or K.quote(sym).get("price")
        if not px:
            print(f"Could not fetch quote for {sym}; skip.")
            continue
        limit = float(px)*(1.0 - limit_slip)  # slightly through the bid
        print(f"[FORCE] Market blocked; retry limit {limit:.8f} on {sym}")
        ok2, info2 = K.place_limit_sell(sym, amt, limit)
        print(("OK" if ok2 else "FAIL"), info2)

if __name__ == "__main__":
    main()
