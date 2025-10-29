#!/usr/bin/env python3
import os, json, sys, pathlib, logging
from typing import Dict, List
import ccxt

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("crypto-live")

STATE_DIR = pathlib.Path(".state")
BAL_FILE = STATE_DIR / "balances.json"
POS_FILE = STATE_DIR / "positions.json"
BUY_PLAN_FILE = STATE_DIR / "buy_plan.json"
ENTRY_FILE = STATE_DIR / "entries.json"
HIGHWATER_FILE = STATE_DIR / "high_water.json"

def ensure_state_dir():
    STATE_DIR.mkdir(parents=True, exist_ok=True)

def read_json(p: pathlib.Path, default):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default

def write_json(p: pathlib.Path, data):
    p.write_text(json.dumps(data, indent=2, sort_keys=True))

def ffloat(x, d=0.0):
    try: return float(x)
    except Exception: return d

def fint(x, d=0):
    try: return int(float(x))
    except Exception: return d

def envb(name, default):
    v = os.getenv(name)
    return (str(v).lower() in ("1","true","yes","on")) if v is not None else default

# ---- knobs
DRY_RUN = os.getenv("DRY_RUN","ON").upper()
MIN_BUY_USD = ffloat(os.getenv("MIN_BUY_USD","10"),10)
MAX_POSITIONS = fint(os.getenv("MAX_POSITIONS","3"),3)
MAX_BUYS_PER_RUN = fint(os.getenv("MAX_BUYS_PER_RUN","0"),0)      # temp: pause buys
UNIVERSE_TOP_K = fint(os.getenv("UNIVERSE_TOP_K","25"),25)
RESERVE_CASH_PCT = ffloat(os.getenv("RESERVE_CASH_PCT","5"),5)
DUST_MIN_USD = ffloat(os.getenv("DUST_MIN_USD","2"),2)
ROTATE_WHEN_FULL = envb("ROTATE_WHEN_FULL", True)
ROTATE_WHEN_CASH_SHORT = envb("ROTATE_WHEN_CASH_SHORT", True)

TAKE_PROFIT_PCT = ffloat(os.getenv("TAKE_PROFIT_PCT","12"),12)
TRAIL_PCT = ffloat(os.getenv("TRAIL_PCT","4"),4)                  # temp: a bit tighter
STOP_LOSS_PCT = ffloat(os.getenv("STOP_LOSS_PCT","10"),10)
MIN_SELL_USD = ffloat(os.getenv("MIN_SELL_USD","10"),10)

KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY","")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET","")

def connects_private(): return bool(KRAKEN_API_KEY and KRAKEN_API_SECRET)
def ex_kraken():
    return ccxt.kraken({"apiKey":KRAKEN_API_KEY,"secret":KRAKEN_API_SECRET,"enableRateLimit":True})

def usd_from_balance(bal: Dict) -> float:
    free = (bal or {}).get("free", {})
    return ffloat(free.get("USD",0)) + ffloat(free.get("ZUSD",0))

def fetch_bal_pos(ex):
    bal = ex.fetch_balance()
    tickers = ex.fetch_tickers()
    prices = {}
    for sym, t in tickers.items():
        if sym.endswith("/USD"):
            prices[sym.split("/")[0]] = ffloat(t.get("last") or t.get("close") or 0.0)
    positions = []
    for coin, amt in bal.get("free",{}).items():
        amount = ffloat(amt,0)
        if amount <= 0: continue
        if coin.upper() in ("USD","ZUSD","USDT","USDC"): continue
        px = prices.get(coin.upper(),0.0)
        est = amount*px if px>0 else 0.0
        positions.append({"asset":coin.upper(),"amount":amount,"est_usd":round(est,2),"px_used":px})
    positions.sort(key=lambda x:x["est_usd"], reverse=True)
    return bal, positions

def symbol_for_asset(ex, asset): 
    sym=f"{asset}/USD"
    m = ex.markets or ex.load_markets()
    return sym if sym in m else ""

def place_market_sell_all(ex, sym, qty):
    if DRY_RUN=="ON": 
        log.info(f"[DRY] SELL {sym} qty={qty}")
        return {"dry":True}
    return ex.create_market_sell_order(sym, qty)

def place_market_buy(ex, sym, qty):
    if DRY_RUN=="ON": 
        log.info(f"[DRY] BUY  {sym} qty={qty}")
        return {"dry":True}
    return ex.create_market_buy_order(sym, qty)

def sell_guard_decision(last_px, entry_px, high_px):
    if last_px<=0 or entry_px<=0: return ""
    pnl = (last_px-entry_px)/entry_px*100.0
    if pnl >= TAKE_PROFIT_PCT: return "TAKE_PROFIT"
    if high_px>0:
        drop = (high_px-last_px)/high_px*100.0
        if drop >= TRAIL_PCT: return "TRAIL"
    if pnl <= -STOP_LOSS_PCT: return "STOP_LOSS"
    return ""

def warm_snapshots():
    ensure_state_dir()
    entries = read_json(ENTRY_FILE, {})
    highs   = read_json(HIGHWATER_FILE, {})
    if connects_private():
        try:
            ex = ex_kraken()
            bal, positions = fetch_bal_pos(ex)
            write_json(BAL_FILE, bal)
            write_json(POS_FILE, {"positions":positions})
            usd = usd_from_balance(bal)

            # --- SEED ENTRIES/HIGHS ONCE if empty (so guard can sell)
            if not entries:
                for p in positions:
                    asset = p["asset"]
                    px = p["px_used"]
                    qty = p["amount"]
                    if px>0 and qty>0:
                        entries[asset] = {"avg_entry": px, "qty": qty}
                        highs[asset] = px
                log.info(f"Seeded entries for {len(entries)} assets from current holdings.")

            # minimal buy plan tonight (buys paused via MAX_BUYS_PER_RUN=0)
            write_json(BUY_PLAN_FILE, {"buy_plan": []})
            write_json(ENTRY_FILE, entries)
            write_json(HIGHWATER_FILE, highs)

            log.info(f"Warm snapshot done. USD={usd:.2f}, positions={len(positions)}, buy_plan=0")
            return
        except Exception as e:
            log.warning(f"Warm snapshot (private) failed: {e}")

    # fallback
    write_json(BAL_FILE, {"free":{"USD":0}})
    write_json(POS_FILE, {"positions":[]})
    write_json(BUY_PLAN_FILE, {"buy_plan":[]})
    write_json(ENTRY_FILE, entries or {})
    write_json(HIGHWATER_FILE, highs or {})
    log.info("Warm snapshot (public-only fallback) written.")

def trade_loop_once():
    ensure_state_dir()
    entries = read_json(ENTRY_FILE, {})
    highs   = read_json(HIGHWATER_FILE, {})

    ex = ex_kraken()
    try:
        bal, positions = fetch_bal_pos(ex)
    except Exception as e:
        log.error(f"Could not fetch balances/positions: {e}")
        bal, positions = {}, []
    usd = usd_from_balance(bal) if bal else 0.0
    write_json(BAL_FILE, bal or {"free":{"USD":usd}})
    write_json(POS_FILE, {"positions":positions})
    log.info(f"USD cash: {usd:.2f} | positions: {len(positions)}")

    # ---- SELL PHASE
    for p in positions:
        asset = p["asset"]; amt = ffloat(p["amount"],0); est = ffloat(p["est_usd"],0)
        if amt<=0 or est<MIN_SELL_USD: 
            continue
        sym = symbol_for_asset(ex, asset)
        if not sym: 
            continue
        last = ffloat(ex.fetch_ticker(sym).get("last"),0)
        ent  = ffloat(entries.get(asset,{}).get("avg_entry",0),0)
        high = max(ffloat(highs.get(asset,0),0), last)
        highs[asset] = high
        decision = sell_guard_decision(last, ent, high)
        if decision:
            log.info(f"[SELL-GUARD] {asset} {decision} | last={last:.6f} entry={ent:.6f} high={high:.6f}")
            try:
                res = place_market_sell_all(ex, sym, amt)
                log.info(f"Placed SELL {sym} qty={amt} result={res}")
                entries.pop(asset, None)
                highs.pop(asset, None)
            except Exception as se:
                log.error(f"SELL failed for {sym}: {se}")

    # ---- BUY PHASE (paused tonight by MAX_BUYS_PER_RUN=0)
    # (kept for completeness; plan remains empty so no buys)

    write_json(ENTRY_FILE, entries)
    write_json(HIGHWATER_FILE, highs)

def main():
    args = sys.argv[1:]
    warm_only = "--warm-only" in args
    loop_once = "--loop-once" in args
    log.info("DRY_RUN=%s | MIN_BUY_USD=%.2f | MAX_POSITIONS=%d | MAX_BUYS_PER_RUN=%d | "
             "TAKE_PROFIT_PCT=%.2f | TRAIL_PCT=%.2f | STOP_LOSS_PCT=%.2f",
             DRY_RUN, MIN_BUY_USD, MAX_POSITIONS, MAX_BUYS_PER_RUN,
             TAKE_PROFIT_PCT, TRAIL_PCT, STOP_LOSS_PCT)
    warm_snapshots()
    if warm_only and not loop_once: return
    trade_loop_once()

if __name__ == "__main__":
    main()
# At end of main.py
from tools.momentum_spike import main as act_on_spikes
print("\n=== Running Momentum Spike Scan ===")
act_on_spikes()

from trader.auto_sell_guard import run_cool_off_guard
print("\n=== Running Auto-Sell Cool-Off Guard ===")
run_cool_off_guard()
