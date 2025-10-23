#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crypto Live — Engine (Kraken, mean-revert buys + SELL LOGIC GUARD)

Adds SELL logic:
- TAKE_PROFIT_PCT: sell when price >= entry*(1+TP)
- STOP_LOSS_PCT:   sell when price <= entry*(1-SL)
- TRAILING_STOP_PCT: after price makes a new peak, sell if it falls by TSL from that peak
  (trailing is only armed once gain >= TRAILING_ARM_PCT)

State:
- .state/spot_positions.json keeps (entry, peak, amount) per symbol to evaluate exits.

Env (wired via workflow repo Variables or defaults here):
  DRY_RUN (ON/OFF)
  KRAKEN_API_KEY / KRAKEN_API_SECRET
  MIN_BUY_USD, MAX_POSITIONS, MAX_BUYS_PER_RUN, UNIVERSE_TOP_K,
  RESERVE_CASH_PCT, ROTATE_WHEN_FULL, ROTATE_WHEN_CASH_SHORT,
  DUST_MIN_USD, DUST_SKIP_STABLES,
  TAKE_PROFIT_PCT (default 3), STOP_LOSS_PCT (default 4),
  TRAILING_STOP_PCT (default 2), TRAILING_ARM_PCT (default 1.5)
"""

from __future__ import annotations
import os, sys, time, math, json, csv, pathlib, traceback, datetime as dt
from typing import Dict, List, Tuple
import ccxt
import pandas as pd

STATE_DIR = pathlib.Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG  = STATE_DIR / "run.log"
KPI_CSV  = STATE_DIR / "kpi_history.csv"
SPEC_TXT = STATE_DIR / "spec_gate_report.txt"
POS_JSON = STATE_DIR / "spot_positions.json"  # entry/peak/amount per symbol

# ---------------- tiny logger ----------------
class Log:
    def __init__(self): self.lines=[]
    def _emit(self, msg): 
        ts = dt.datetime.utcnow().isoformat()+"+00:00"
        line=f"[{ts}] {msg}"
        print(line, flush=True); self.lines.append(line)
    def info(self,m): self._emit(m)
    def warn(self,m): self._emit("warn: "+m)
    def error(self,m): self._emit("ERROR: "+m)
    def buy(self,pair,px): self._emit(f"[BUY] {pair} @ {px:.5f}")
    def sell(self,pair,px,reason): self._emit(f"[SELL] {pair} @ {px:.5f} reason={reason}")
    def summary(self,buys,sells,open_,dry): self._emit(f"SUMMARY buys={buys} sells={sells} open={open_} DRY_RUN={dry}")

log = Log()

# ---------------- env helpers ----------------
def env_str(n,d=""): return str(os.getenv(n,d))
def env_bool(n,d="false"): return env_str(n,d).strip().lower() in ("1","true","on","yes","y")
def env_float(n,d="0"): 
    try: return float(env_str(n,d))
    except: return float(d)
def env_int(n,d="0"):
    try: return int(float(env_str(n,d)))
    except: return int(float(d))

# ---------------- config ----------------
DRY_RUN       = env_str("DRY_RUN","ON").upper()=="ON"
MIN_BUY_USD   = env_float("MIN_BUY_USD","15")
MAX_POSITIONS = env_int("MAX_POSITIONS","3")
MAX_BUYS_PER_RUN = env_int("MAX_BUYS_PER_RUN","1")
UNIVERSE_TOP_K   = env_int("UNIVERSE_TOP_K","25")
RESERVE_CASH_PCT = env_float("RESERVE_CASH_PCT","5")
ROTATE_WHEN_FULL       = env_bool("ROTATE_WHEN_FULL","true")
ROTATE_WHEN_CASH_SHORT = env_bool("ROTATE_WHEN_CASH_SHORT","true")
DUST_MIN_USD    = env_float("DUST_MIN_USD","2")
DUST_SKIP_STABLES = env_bool("DUST_SKIP_STABLES","true")

# --- SELL LOGIC knobs (add these as repo Variables later if you want) ---
TAKE_PROFIT_PCT    = env_float("TAKE_PROFIT_PCT","3.0")     # <<< TAKE_PROFIT keyword for guard
STOP_LOSS_PCT      = env_float("STOP_LOSS_PCT","4.0")       # <<< STOP_LOSS keyword for guard
TRAILING_STOP_PCT  = env_float("TRAILING_STOP_PCT","2.0")   # <<< trailing keyword for guard
TRAILING_ARM_PCT   = env_float("TRAILING_ARM_PCT","1.5")    # gain needed to arm trailing stop

API_KEY = env_str("KRAKEN_API_KEY","")
API_SEC = env_str("KRAKEN_API_SECRET","")
if not API_KEY or not API_SEC:
    raise RuntimeError("Kraken credentials missing: set KRAKEN_API_KEY and KRAKEN_API_SECRET.")

kraken = ccxt.kraken({"apiKey":API_KEY, "secret":API_SEC, "enableRateLimit":True})

# ---------------- utils ----------------
USD_STABLE_CODES = {"USDT","USDC","DAI","TUSD","FDUSD","USDP","GUSD","PYUSD"}

def write_artifacts():
    try: RUN_LOG.write_text("\n".join(log.lines), encoding="utf-8")
    except: pass

def append_kpi_csv(bal_usd: float):
    header=["timestamp_utc","bal_usd"]
    is_new = not KPI_CSV.exists()
    with KPI_CSV.open("a", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        if is_new: w.writerow(header)
        w.writerow([dt.datetime.utcnow().isoformat()+"+00:00", f"{bal_usd:.2f}"])

def write_spec(txt:str): SPEC_TXT.write_text(txt, encoding="utf-8")

def load_positions() -> Dict[str,dict]:
    if POS_JSON.exists():
        try: return json.loads(POS_JSON.read_text())
        except: return {}
    return {}
def save_positions(d:Dict[str,dict]): POS_JSON.write_text(json.dumps(d, indent=2))

def safe_get(d,*keys,default=None):
    cur=d
    for k in keys:
        if not isinstance(cur,dict) or k not in cur: return default
        cur=cur[k]
    return cur

# ---------------- market + balances ----------------
def fetch_balance_raw() -> Dict[str,float]:
    try:
        b = kraken.fetch_balance()
        return {k: float(v or 0) for k,v in b.get("total",{}).items()}
    except Exception as e:
        log.warn(f'fetch_balance failed: {e}')
        return {}

def usd_cash_from_raw(raw:Dict[str,float]) -> float:
    return float(raw.get("USD",0.0) + raw.get("ZUSD",0.0))

def list_usd_pairs(limit:int=80) -> List[str]:
    markets = kraken.load_markets()
    out=[]
    for s,m in markets.items():
        if not m.get("active",True): continue
        if m.get("quote","").upper()!="USD": continue
        base=m.get("base","").upper()
        if DUST_SKIP_STABLES and base in USD_STABLE_CODES: continue
        out.append(s)
    return out[:max(1,limit)]

def ohlcv_change(pair:str, timeframe="15m", candles=20) -> Tuple[float,float]:
    try:
        o = kraken.fetch_ohlcv(pair, timeframe=timeframe, limit=candles)
        if not o or len(o)<3: return (math.nan, float("-inf"))
        last=o[-1][4]; prev=o[-3][4]
        if prev>0: return (float(last), (last-prev)/prev)
        return (float(last), float("-inf"))
    except: return (math.nan, float("-inf"))

def best_bid_ask(pair:str) -> Tuple[float,float]:
    try:
        t = kraken.fetch_ticker(pair)
        bid = safe_get(t,"bid", default=None)
        ask = safe_get(t,"ask", default=None)
        if not ask or ask<=0: ask = safe_get(t,"last", default=None)
        return float(bid or 0), float(ask or 0)
    except: return (0.0,0.0)

# ---------------- trade ops ----------------
def place_order_with_log(symbol:str, side:str, amount:float):
    try:
        resp = kraken.create_order(symbol=symbol, type="market", side=side, amount=amount)
        txid = safe_get(resp, "id", default=None)
        log.info(f'EXCHANGE OK AddOrder accepted | txid={txid} | side={side} symbol={symbol} amount={amount}')
        return resp
    except Exception as e:
        log.error(f'EXCHANGE_ERROR AddOrder exception: {e} | side={side} symbol={symbol} amount={amount}')
        raise

# ---------------- strategy helpers ----------------
def pick_candidates(pairs:List[str], top_k:int) -> List[Tuple[str,float,float]]:
    rows=[]
    for p in pairs:
        last, ch = ohlcv_change(p)
        if not last or not math.isfinite(ch): continue
        rows.append((p,last,ch))
    rows.sort(key=lambda r:r[2])  # most negative first (dip)
    return rows[:max(1,top_k)]

def can_afford_buy(usd_free:float) -> bool:
    reserve = RESERVE_CASH_PCT/100.0
    spend_cap = usd_free*(1-reserve)
    return spend_cap >= MIN_BUY_USD + 0.01

def symbol_for_base(base:str, markets:Dict[str,dict]) -> str|None:
    for s,m in markets.items():
        if m.get("base","").upper()==base.upper() and m.get("quote","").upper()=="USD" and m.get("active",True):
            return s
    return None

# ---------------- SELL RULES ----------------
def check_sell_rules(symbol:str, price:float, pos:dict) -> str|None:
    """
    Returns reason string if we should sell now, else None.
    Implements:
      - TAKE_PROFIT (>= TAKE_PROFIT_PCT)
      - STOP_LOSS (<= STOP_LOSS_PCT)
      - trailing stop (armed after TRAILING_ARM_PCT gain; drop >= TRAILING_STOP_PCT from peak)
    """
    entry = float(pos.get("entry",0))
    peak  = float(pos.get("peak",entry))
    if entry <= 0 or price <= 0:
        return None

    gain = (price/entry - 1.0) * 100.0
    dd_from_peak = (1.0 - price/max(peak, entry)) * 100.0

    # TAKE_PROFIT
    if gain >= TAKE_PROFIT_PCT:
        return f"TAKE_PROFIT +{gain:.2f}% ≥ {TAKE_PROFIT_PCT:.2f}%"

    # STOP_LOSS
    if gain <= -abs(STOP_LOSS_PCT):
        return f"STOP_LOSS {gain:.2f}% ≤ -{abs(STOP_LOSS_PCT):.2f}%"

    # trailing (only if armed)
    if gain >= TRAILING_ARM_PCT and dd_from_peak >= TRAILING_STOP_PCT:
        return f"TRAILING_DROP {dd_from_peak:.2f}% ≥ {TRAILING_STOP_PCT:.2f}% (armed at +{TRAILING_ARM_PCT:.2f}%)"

    return None

# ---------------- main ----------------
def main():
    buys = sells = 0
    spec = [
        f"DRY_RUN={DRY_RUN}",
        f"MIN_BUY_USD={MIN_BUY_USD}",
        f"MAX_POSITIONS={MAX_POSITIONS} MAX_BUYS_PER_RUN={MAX_BUYS_PER_RUN}",
        f"UNIVERSE_TOP_K={UNIVERSE_TOP_K} RESERVE_CASH_PCT={RESERVE_CASH_PCT}",
        f"ROTATE_WHEN_FULL={ROTATE_WHEN_FULL} ROTATE_WHEN_CASH_SHORT={ROTATE_WHEN_CASH_SHORT}",
        f"DUST_MIN_USD={DUST_MIN_USD} DUST_SKIP_STABLES={DUST_SKIP_STABLES}",
        f"TAKE_PROFIT_PCT={TAKE_PROFIT_PCT} STOP_LOSS_PCT={STOP_LOSS_PCT} TRAILING_STOP_PCT={TRAILING_STOP_PCT} TRAILING_ARM_PCT={TRAILING_ARM_PCT}",
    ]

    markets = kraken.load_markets()
    usd_pairs = list_usd_pairs(limit=80)

    raw_bal = fetch_balance_raw()
    usd_cash = usd_cash_from_raw(raw_bal)

    # Load per-symbol position state
    pos_state = load_positions()

    # --- SELL PASS over existing holdings ---
    # Determine open positions (value >= dust)
    open_symbols=[]
    for base, amt in raw_bal.items():
        bu = base.upper()
        if bu in ("USD","ZUSD"): continue
        sym = symbol_for_base(bu, markets)
        if not sym: continue
        _, ask = best_bid_ask(sym)
        value = float(amt) * float(ask or 0)
        if value >= max(2.0, DUST_MIN_USD):
            open_symbols.append((sym, bu, float(amt), float(ask or 0)))

    # Evaluate exits
    for sym, base, amt, price in open_symbols:
        # init state if missing
        st = pos_state.get(sym, {"entry": price, "peak": price, "amount": amt})
        # keep amount synced if it grew/shrank
        st["amount"] = amt
        # update peak
        if price > st.get("peak", price): st["peak"] = price
        pos_state[sym] = st

        reason = check_sell_rules(sym, price, st)
        if reason and not DRY_RUN:
            try:
                place_order_with_log(sym, "sell", amt)
                log.sell(sym, price, reason)
                sells += 1
                # remove from state after sell
                pos_state.pop(sym, None)
            except Exception as e:
                log.warn(f"sell failed {sym}: {e}")
        elif reason and DRY_RUN:
            log.sell(sym, price, reason + " (DRY_RUN)")
            sells += 1
            pos_state.pop(sym, None)

    # Refresh balances after potential sells
    raw_bal = fetch_balance_raw()
    usd_cash = usd_cash_from_raw(raw_bal)

    # Count open positions after sells (ignore dust)
    opens_after = 0
    for base, amt in raw_bal.items():
        bu=base.upper()
        if bu in ("USD","ZUSD"): continue
        sym = symbol_for_base(bu, markets)
        if not sym: continue
        _, ask = best_bid_ask(sym)
        if float(amt)*float(ask or 0) >= max(2.0, DUST_MIN_USD):
            opens_after += 1

    # --- BUY PASS (mean-revert) ---
    candidates = pick_candidates(usd_pairs, UNIVERSE_TOP_K)
    buys_allowed = max(0, MAX_BUYS_PER_RUN)

    rotate = False
    if opens_after >= MAX_POSITIONS and ROTATE_WHEN_FULL:
        rotate = True; spec.append("Rotation reason: FULL")
    if not can_afford_buy(usd_cash) and ROTATE_WHEN_CASH_SHORT:
        rotate = True; spec.append("Rotation reason: CASH_SHORT")

    # (optional) very light rotation: already handled by SELL stage via rules; skip extra churn

    for pair, last, ch in candidates:
        if buys >= buys_allowed: break
        if opens_after >= MAX_POSITIONS: break
        _, ask = best_bid_ask(pair)
        px = ask or last
        if not px or px <= 0: continue
        if not can_afford_buy(usd_cash): break
        amount = float(f"{(MIN_BUY_USD/px):.8f}")
        if amount <= 0: continue

        if DRY_RUN:
            log.buy(pair, px)
            buys += 1; opens_after += 1; usd_cash -= MIN_BUY_USD
            # set entry/peak in state (simulate)
            pos_state[pair] = {"entry": px, "peak": px, "amount": amount}
        else:
            try:
                place_order_with_log(pair, "buy", amount)
                log.buy(pair, px)
                buys += 1; opens_after += 1; usd_cash -= MIN_BUY_USD
                # initialize state for this symbol
                pos_state[pair] = {"entry": px, "peak": px, "amount": amount}
            except Exception as e:
                log.warn(f"buy failed {pair}: {e}")

    # KPI: current USD balance snapshot (best-effort)
    try:
        bal_now = usd_cash_from_raw(fetch_balance_raw())
    except:
        bal_now = usd_cash
    try: append_kpi_csv(bal_now)
    except: pass

    # Save state/artifacts
    save_positions(pos_state)
    log.summary(buys, sells, opens_after, DRY_RUN)
    write_artifacts()
    write_spec("\n".join(spec))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"fatal: {e}\n{traceback.format_exc()}")
        write_artifacts()
        raise
