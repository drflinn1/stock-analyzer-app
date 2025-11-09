#!/usr/bin/env python3
"""
Portfolio snapshot: appends a row to .state/portfolio_history.csv each run.

Columns:
when_utc,total_usd,fiat_usd,stable_usd,position_pair,position_amt,position_px,position_val_usd,notes
"""
from __future__ import annotations
import csv, json, os, time
from pathlib import Path
from typing import Optional, Tuple
import ccxt

STATE = Path(".state"); STATE.mkdir(parents=True, exist_ok=True)
HIST  = STATE / "portfolio_history.csv"
POS   = STATE / "positions.json"

def _kraken():
    key = os.getenv("KRAKEN_API_KEY") or os.getenv("KRAKEN_KEY") or ""
    sec = os.getenv("KRAKEN_API_SECRET") or os.getenv("KRAKEN_SECRET") or ""
    ex = ccxt.kraken({"apiKey": key, "secret": sec, "enableRateLimit": True})
    ex.load_markets()
    return ex

def _read_pos() -> Tuple[Optional[str], float]:
    if not POS.exists(): return None, 0.0
    try:
        d = json.loads(POS.read_text())
        return d.get("pair"), float(d.get("amount",0))
    except Exception:
        return None, 0.0

def _px(ex, pair: str) -> Optional[float]:
    try:
        t = ex.fetch_ticker(pair)
        p = t.get("last") or t.get("close") or t.get("bid") or t.get("ask")
        return float(p) if p else None
    except Exception:
        return None

def snapshot() -> None:
    ex = _kraken()
    balances = ex.fetch_balance().get("total", {})

    # Treat fiat/stables at $1
    usd  = float(balances.get("USD", 0.0))
    usdc = float(balances.get("USDC", 0.0))
    usdt = float(balances.get("USDT", 0.0))
    stable_usd = usdc + usdt
    fiat_usd = usd

    pair, amt = _read_pos()
    px = val = 0.0
    note = ""
    if pair and amt > 0:
        p = _px(ex, pair)
        if p: 
            px = p; val = amt * p
        else:
            note = f"no_px:{pair}"

    total = fiat_usd + stable_usd + val

    write_header = not HIST.exists()
    with HIST.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["when_utc","total_usd","fiat_usd","stable_usd","position_pair","position_amt","position_px","position_val_usd","notes"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), 
                    f"{total:.4f}", f"{fiat_usd:.4f}", f"{stable_usd:.4f}",
                    pair or "", f"{amt:.8f}", f"{px:.8f}", f"{val:.4f}", note])

if __name__ == "__main__":
    snapshot()
