# -*- coding: utf-8 -*-
"""
Crypto engine with tight SELL GUARD + cost-basis backfill from trade history.
Auto-sizes BUY orders to satisfy Kraken min cost/amount/precision.

SELL knobs (Variables):
  SELL_HARD_STOP_PCT=3
  SELL_TRAIL_PCT=2
  SELL_TAKE_PROFIT_PCT=5
  BACKFILL_LOOKBACK_DAYS=60

Other knobs (unchanged): DRY_RUN, MIN_BUY_USD, MAX_BUYS_PER_RUN, MAX_POSITIONS,
RESERVE_CASH_PCT, UNIVERSE_TOP_K, MIN_24H_PCT, MIN_BASE_VOL_USD, WHITELIST, etc.

Secrets: KRAKEN_API_KEY, KRAKEN_API_SECRET
"""

import os, json, time, math
from pathlib import Path
from typing import Dict, List, Tuple

import ccxt

STATE_DIR = Path(".state"); STATE_DIR.mkdir(parents=True, exist_ok=True)
LEDGER = STATE_DIR / "positions.json"
BUY_GATES_MD = STATE_DIR / "buy_gates.md"

STABLES = {"USD","USDT","USDC","EUR","GBP"}
EXCLUDE_TICKERS = {"SPX","PUMP","BABY","ALKIMI"}

def _now(): return time.strftime("%Y-%m-%d %H:%M:%S")
def _b(s, d): return str(s).lower() in {"1","true","yes","on"} if s is not None else d
def _fi(name, default):
    try: return float(os.getenv(name, default))
    except: return float(default)
def _ii(name, default):
    try: return int(float(os.getenv(name, default)))
    except: return int(default)

def build_exchange(api_key:str, api_secret:str) -> ccxt.Exchange:
    ex = ccxt.kraken({
        "apiKey": api_key or "",
        "secret": api_secret or "",
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    })
    ex.load_markets()
    return ex

def _is_bad_coin(symbol:str) -> bool:
    if "/" not in symbol: return True
    base, quote = symbol.split("/")
    if base.upper() in EXCLUDE_TICKERS: return True
    if quote.upper() != "USD": return True
    if base.upper() in STABLES: return True
    return False

def pick_candidates(ex: ccxt.Exchange, top_k:int) -> List[Dict]:
    tickers = ex.fetch_tickers()
    out=[]
    for sym,t in tickers.items():
        if not sym.endswith("/USD"): continue
        if _is_bad_coin(sym): continue
        ch = t.get("percentage")
        if ch is None:
            last = t.get("last") or t.get("close"); op = t.get("open")
            if last and op: ch = (last-op)/op*100.0
            else: continue
        qv = float(t.get("quoteVolume") or 0.0)
        if qv < 10000: continue
        out.append({"symbol":sym,"price":float(t.get("last") or t.get("close") or 0.0),
                    "change24h":float(ch),"quoteVolUsd":qv})
    out.sort(key=lambda r:(r["change24h"], r["quoteVolUsd"]), reverse=True)
    return out[:max(1,int(top_k))]

def fetch_positions_snapshot(ex: ccxt.Exchange):
    balances = ex.fetch_balance()
    prices = ex.fetch_tickers()
    pos=[]
    for sym,m in ex.markets.items():
        if _is_bad_coin(sym) or not sym.endswith("/USD"): continue
        base=m["base"]
        qty = float((balances.get(base,{}) or {}).get("total") or 0.0)
        if qty<=0: continue
        price = float((prices.get(sym,{}) or {}).get("last") or 0.0)
        pos.append({"symbol":sym,"base":base,"qty":qty,"price":price,"usd_value":qty*price})
    return pos

def get_cash_balance_usd(ex: ccxt.Exchange) -> float:
    bal = ex.fetch_balance()
    usd = bal.get("USD",{})
    free = float(usd.get("free") or 0.0)
    total = float(usd.get("total") or free)
    return max(free,total)

# ---------- Sizing helpers (fixes "volume minimum not met") ----------
def _market_limits(m: dict) -> Tuple[float,float,float]:
    """Return (min_cost, min_amount, amount_step) if available."""
    lims = m.get("limits", {}) if m else {}
    cost_min   = float((lims.get("cost",   {}) or {}).get("min") or 0.0)
    amount_min = float((lims.get("amount", {}) or {}).get("min") or 0.0)
    # amount "step" can be in limits.amount.step or in precision.amount (as decimals)
    step = float((lims.get("amount", {}) or {}).get("step") or 0.0)
    if step <= 0:
        prec = (m.get("precision", {}) or {}).get("amount")
        if isinstance(prec, int) and prec >= 0:
            step = 10 ** (-prec) if prec > 0 else 1.0
    if step <= 0:
        step = 0.0
    return cost_min, amount_min, step

def _ceil_to_step(x: float, step: float) -> float:
    if step and step > 0:
        return math.ceil(x / step) * step
    return x

def _ensure_notional_and_qty(ex: ccxt.Exchange, symbol: str, desired_notional: float) -> Tuple[float, float, float]:
    """
    Given desired spend in USD, return (final_notional, qty, last_price)
    that satisfies Kraken min cost + min amount + precision.
    """
    m = ex.markets[symbol]
    last = float(ex.fetch_ticker(symbol)["last"] or 0.0)
    if last <= 0:
        raise ValueError(f"Ticker has no last price for {symbol}")

    cost_min, amt_min, step = _market_limits(m)

    # Enforce: at least $25 (universal floor), min cost, and price * min amount
    floor_usd = 25.0
    needed_by_amt = amt_min * last if amt_min > 0 else 0.0
    notional = max(desired_notional, cost_min, needed_by_amt, floor_usd)

    # Compute qty from notional, then enforce min amount and step
    qty = notional / max(last, 1e-9)
    qty = max(qty, amt_min)
    qty = _ceil_to_step(qty, step)
    qty = float(ex.amount_to_precision(symbol, qty))

    # Recompute notional after precision/step bump
    notional = qty * last

    print(f"[sizing] {symbol} last={last:.8f} cost_min={cost_min:.4f} amt_min={amt_min} step={step} "
          f"req_notional>={desired_notional:.2f} => buy ${notional:.2f} qty≈{qty}")
    return notional, qty, last
# ---------------------------------------------------------------------

def buy_market(ex: ccxt.Exchange, symbol:str, notional:float, dry:bool):
    notional, qty, last = _ensure_notional_and_qty(ex, symbol, notional)
    if dry:
        print(f"[order][dry-run] BUY {symbol} ${notional:.2f} qty≈{qty}")
        return {"id":"dry","symbol":symbol,"amount":float(qty),"price":last}
    print(f"[order] BUY {symbol} ${notional:.2f} qty≈{qty}")
    return ex.create_market_buy_order(symbol, float(qty))

def sell_market(ex: ccxt.Exchange, symbol:str, qty:float, dry:bool):
    m = ex.markets[symbol]
    _, amt_min, step = _market_limits(m)
    qty = max(qty, amt_min)
    qty = _ceil_to_step(qty, step)
    qp = float(ex.amount_to_precision(symbol, qty))
    if dry:
        print(f"[order][dry-run] SELL {symbol} qty≈{qp}")
        return {"id":"dry","symbol":symbol,"amount":qp}
    print(f"[order] SELL {symbol} qty≈{qp}")
    return ex.create_market_sell_order(symbol, qp)

# ----- ledger -----
def load_ledger()->Dict[str,Dict]:
    if LEDGER.exists():
        try: return json.loads(LEDGER.read_text() or "{}")
        except: return {}
    return {}

def save_ledger(d:Dict[str,Dict])->None:
    LEDGER.write_text(json.dumps(d, indent=2))

def ensure_entry(ledger:Dict[str,Dict], symbol:str, entry:float, set_high=True):
    rec = ledger.get(symbol.upper())
    if not rec:
        ledger[symbol.upper()] = {"entry":float(entry), "high":float(entry if set_high else 0.0), "added":_now()}
    else:
        rec["entry"] = float(rec.get("entry", entry))
        if set_high:
            rec["high"] = float(max(rec.get("high", entry), entry))

def update_high(ledger:Dict[str,Dict], symbol:str, price:float):
    rec=ledger.get(symbol.upper())
    if not rec:
        ledger[symbol.upper()]={"entry":float(price),"high":float(price),"added":_now()}
    else:
        rec["high"]=float(max(rec.get("high",price), price))

# ----- cost-basis backfill from trades -----
def _backfill_avg_entry_from_trades(ex:ccxt.Exchange, symbol:str, current_qty:float, lookback_days:int)->float or None:
    try:
        since = int(time.time()*1000) - lookback_days*24*3600*1000
        qty_net=0.0; cost_usd=0.0
        cursor_since = since
        for _ in range(2):
            trades = ex.fetch_my_trades(symbol=symbol, since=cursor_since, limit=200)
            if not trades: break
            for t in trades:
                side = (t.get("side") or "").lower()
                price = float(t.get("price") or 0.0)
                amount = float(t.get("amount") or 0.0)
                if price<=0 or amount<=0: continue
                if side=="buy":
                    qty_net += amount
                    cost_usd += amount*price
                elif side=="sell":
                    avg = cost_usd/max(qty_net,1e-9) if qty_net>0 else price
                    reduce = min(qty_net, amount)
                    qty_net -= reduce
                    cost_usd -= reduce*avg
            cursor_since = int(trades[-1]["timestamp"])+1 if trades else cursor_since
        if qty_net<=0: return None
        avg_entry = cost_usd/max(qty_net,1e-9)
        return float(avg_entry)
    except Exception as e:
        print(f"[backfill] {symbol} failed: {e}")
        return None

# ----- main loop -----
def run_trading_loop()->int:
    DRY_RUN = os.getenv("DRY_RUN","ON").upper()
    dry = DRY_RUN!="OFF"
    api_key = os.getenv("KRAKEN_API_KEY",""); api_secret=os.getenv("KRAKEN_API_SECRET","")

    MIN_BUY_USD     = _fi("MIN_BUY_USD",10.0)
    MAX_BUYS_PER_RUN= _ii("MAX_BUYS_PER_RUN",2)
    MAX_POSITIONS   = _ii("MAX_POSITIONS",6)
    RESERVE_CASH_PCT= _fi("RESERVE_CASH_PCT",0.0)

    UNIVERSE_TOP_K  = _ii("UNIVERSE_TOP_K",25)
    MIN_24H_PCT     = _fi("MIN_24H_PCT",0.0)
    MIN_BASE_VOL_USD= _fi("MIN_BASE_VOL_USD",10000.0)

    SELL_HARD_STOP_PCT   = _fi("SELL_HARD_STOP_PCT",3.0)
    SELL_TRAIL_PCT       = _fi("SELL_TRAIL_PCT",2.0)
    SELL_TAKE_PROFIT_PCT = _fi("SELL_TAKE_PROFIT_PCT",5.0)
    LOOKBACK_DAYS        = _ii("BACKFILL_LOOKBACK_DAYS",60)

    WL = [s.strip().upper() for s in (os.getenv("WHITELIST","") or "").split(",") if s.strip()]

    ex = build_exchange(api_key, api_secret)

    ledger = load_ledger()
    positions = fetch_positions_snapshot(ex)
    held = {p["symbol"].upper() for p in positions}

    # --- Backfill entries for existing bags so sells can trigger immediately
    for p in positions:
        sym = p["symbol"].upper()
        if sym not in ledger or ledger[sym].get("entry") is None:
            avg = _backfill_avg_entry_from_trades(ex, sym, p["qty"], LOOKBACK_DAYS)
            if avg and avg>0:
                ensure_entry(ledger, sym, avg, set_high=True)
                print(f"[backfill] {sym} entry≈{avg:.6f}")
            else:
                ensure_entry(ledger, sym, p["price"] or 0.0, set_high=True)
    save_ledger(ledger)

    # --- SELL GUARD (tight)
    sells=0
    for p in positions:
        sym=p["symbol"].upper(); price=float(p["price"] or 0.0); qty=float(p["qty"] or 0.0)
        if qty<=0 or price<=0: continue
        update_high(ledger, sym, price)
        rec = ledger.get(sym,{})
        entry=float(rec.get("entry", price)); high=float(rec.get("high", price))
        ch_entry = (price/entry-1.0)*100.0 if entry>0 else 0.0
        dd_high  = (price/high-1.0)*100.0 if high>0 else 0.0

        if ch_entry >= SELL_TAKE_PROFIT_PCT:
            print(f"[SELL] TAKE_PROFIT {sym} @ {price:.6f} (+{ch_entry:.2f}%)")
            sell_market(ex, sym, qty, dry); sells+=1; ledger.pop(sym,None); continue
        if dd_high <= -abs(SELL_TRAIL_PCT):
            print(f"[SELL] TRAIL {sym} @ {price:.6f} (draw {dd_high:.2f}% from high {high:.6f})")
            sell_market(ex, sym, qty, dry); sells+=1; ledger.pop(sym,None); continue
        if ch_entry <= -abs(SELL_HARD_STOP_PCT):
            print(f"[SELL] STOP_LOSS {sym} @ {price:.6f} ({ch_entry:.2f}% vs entry {entry:.6f})")
            sell_market(ex, sym, qty, dry); sells+=1; ledger.pop(sym,None); continue
    save_ledger(ledger)

    # refresh after sells
    if sells:
        positions = fetch_positions_snapshot(ex)
        held = {p["symbol"].upper() for p in positions}

    # --- BUY side
    cash = get_cash_balance_usd(ex)
    spendable = max(0.0, cash - cash*(RESERVE_CASH_PCT/100.0))
    universe = pick_candidates(ex, UNIVERSE_TOP_K)
    if WL: universe = [c for c in universe if c["symbol"].upper() in WL]
    filtered = [c for c in universe if c["change24h"]>=MIN_24H_PCT and c["quoteVolUsd"]>=MIN_BASE_VOL_USD and c["symbol"].upper() not in held]

    BUY_GATES_MD.write_text("\n".join([
        "## BUY_DIAGNOSTIC",
        f"- time: {_now()}",
        f"- cash: {cash:.2f} spendable: {spendable:.2f}",
        f"- positions: {len(held)}",
        f"- candidates: total={len(universe)} after_filters={len(filtered)}",
        f"- whitelist: {'on' if WL else 'off'}",
    ]) + "\n")

    buys=0
    max_new = max(0, MAX_POSITIONS - len(held))
    slots = min(MAX_BUYS_PER_RUN, max_new)
    for c in filtered:
        if buys>=slots or spendable<MIN_BUY_USD: break
        sym=c["symbol"].upper()
        try:
            buy_market(ex, sym, MIN_BUY_USD, dry)
            last=float(ex.fetch_ticker(sym)["last"])
            ensure_entry(ledger, sym, last, set_high=True); save_ledger(ledger)
            buys+=1; spendable-=MIN_BUY_USD
        except Exception as e:
            print(f"[engine] BUY failed for {sym}: {e}")

    if buys==0 and sells==0:
        print("[engine] No buys executed this run.")
    else:
        if buys:  print(f"[engine] Buys executed: {buys}")
        if sells: print(f"[engine] Sells executed: {sells}")
    return 0

if __name__ == "__main__":
    raise SystemExit(run_trading_loop())

# ---- adapter for new main.py API ----
def run_hourly_rotation(
    dry_run: bool = True,
    buy_usd: float = 25.0,
    tp_pct: float = 5.0,
    stop_pct: float = 1.0,
    window_min: int = 60,
    slow_gain_req: float = 3.0,
    universe_pick: str | None = None,
) -> int:
    """Compatibility wrapper so main.py can call this symbol."""
    return run_trading_loop()
