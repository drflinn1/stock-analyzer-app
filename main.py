#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crypto Live — Engine (Kraken) with:
- Mean-revert buys
- TAKE_PROFIT / STOP_LOSS / TRAILING exits
- Restricted-market auto-blacklist (EAccount:Invalid permissions)
- Toggle: SKIP_RESTRICTED (default True)

Artifacts written under .state/:
  - run.log
  - kpi_history.csv
  - spec_gate_report.txt
  - spot_positions.json        (entry/peak/amount per symbol)
  - restricted_markets.json    (persisted blacklist)
"""

from __future__ import annotations
import os, sys, time, math, json, csv, pathlib, traceback, datetime as dt
from typing import Dict, List, Tuple
import ccxt
import pandas as pd

# ---------------- paths ----------------
STATE_DIR = pathlib.Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG  = STATE_DIR / "run.log"
KPI_CSV  = STATE_DIR / "kpi_history.csv"
SPEC_TXT = STATE_DIR / "spec_gate_report.txt"
POS_JSON = STATE_DIR / "spot_positions.json"
RESTRICTED_JSON = STATE_DIR / "restricted_markets.json"

# ---------------- tiny logger ----------------
class Log:
    def __init__(self): self.lines=[]
    def _emit(self, msg):
        ts = dt.datetime.utcnow().isoformat()+"+00:00"
        line = f"[{ts}] {msg}"
        print(line, flush=True); self.lines.append(line)
    def info(self,m): self._emit(m)
    def warn(self,m): self._emit("warn: "+m)
    def error(self,m): self._emit("ERROR: "+m)
    def buy(self,pair,px): self._emit(f"[BUY] {pair} @ {px:.5f}")
    def sell(self,pair,px,reason): self._emit(f"[SELL] {pair} @ {px:.5f} reason={reason}")
    def skipped(self,pair,reason): self._emit(f"SKIPPED_RESTRICTED {pair} reason={reason}")
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

# SELL logic
TAKE_PROFIT_PCT    = env_float("TAKE_PROFIT_PCT","3.0")     # TAKE_PROFIT keyword
STOP_LOSS_PCT      = env_float("STOP_LOSS_PCT","4.0")       # STOP_LOSS keyword
TRAILING_STOP_PCT  = env_float("TRAILING_STOP_PCT","2.0")   # trailing keyword
TRAILING_ARM_PCT   = env_float("TRAILING_ARM_PCT","1.5")

# Toggle for restricted-market avoidance
SKIP_RESTRICTED = env_bool("SKIP_RESTRICTED","true")

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

def load_json(path: pathlib.Path, default):
    if path.exists():
        try: return json.loads(path.read_text())
        except: return default
    return default

def save_json(path: pathlib.Path, obj):
    path.write_text(json.dumps(obj, indent=2))

def load_positions() -> Dict[str,dict]:
    return load_json(POS_JSON, {})

def save_positions(d:Dict[str,dict]):
    save_json(POS_JSON, d)

def load_restricted() -> Dict[str,bool]:
    # stored as { "BLESS/USD": true, ... }
    return load_json(RESTRICTED_JSON, {})

def save_restricted(d:Dict[str,bool]):
    save_json(RESTRICTED_JSON, d)

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

def is_restricted_symbol(symbol: str, restricted: Dict[str,bool]) -> bool:
    return SKIP_RESTRICTED and restricted.get(symbol, False)

def mark_restricted(symbol: str, restricted: Dict[str,bool], reason: str):
    if not restricted.get(symbol):
        restricted[symbol] = True
        save_restricted(restricted)
    log.skipped(symbol, reason)

# ---------------- trade ops ----------------
def place_order_with_log(symbol:str, side:str, amount:float, restricted: Dict[str,bool]):
    """Create a market order; if Kraken rejects due to jurisdiction/permissions, blacklist symbol and return None."""
    try:
        resp = kraken.create_order(symbol=symbol, type="market", side=side, amount=amount)
        txid = safe_get(resp, "id", default=None)
        log.info(f'EXCHANGE OK AddOrder accepted | txid={txid} | side={side} symbol={symbol} amount={amount}')
        return resp
    except Exception as e:
        emsg = str(e)
        # Kraken will reply e.g. 'EAccount:Invalid permissions:BLESS trading restricted for US:WA.'
        # Catch-and-blacklist on those—no crash.
        if "Invalid permissions" in emsg or "restricted for" in emsg or "EAccount" in emsg:
            mark_restricted(symbol, restricted, emsg)
            return None
        # other errors bubble but won’t stop the whole job
        log.warn(f'order failed {symbol}: {e}')
        return None

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

    # trailing (armed)
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
        f"SKIP_RESTRICTED={SKIP_RESTRICTED}",
    ]

    markets = kraken.load_markets()
    usd_pairs = list_usd_pairs(limit=80)
    restricted = load_restricted()  # persisted across runs

    # Remove restricted markets up front to avoid churn
    if SKIP_RESTRICTED:
        usd_pairs = [s for s in usd_pairs if not is_restricted_symbol(s, restricted)]

    raw_bal = fetch_balance_raw()
    usd_cash = usd_cash_from_raw(raw_bal)

    pos_state = load_positions()

    # --- SELL pass ---
    open_symbols=[]
    for base, amt in raw_bal.items():
        bu = base.upper()
        if bu in ("USD","ZUSD"): continue
        sym = symbol_for_base(bu, markets)
        if not sym: continue
        if is_restricted_symbol(sym, restricted):
            # If we hold a restricted asset, we won't attempt to trade it; just skip.
            log.skipped(sym, "present in holdings but restricted for jurisdiction")
            continue
        _, ask = best_bid_ask(sym)
        value = float(amt) * float(ask or 0)
        if value >= max(2.0, DUST_MIN_USD):
            open_symbols.append((sym, bu, float(amt), float(ask or 0)))

    for sym, base, amt, price in open_symbols:
        st = pos_state.get(sym, {"entry": price, "peak": price, "amount": amt})
        st["amount"] = amt
        if price > st.get("peak", price): st["peak"] = price
        pos_state[sym] = st

        reason = check_sell_rules(sym, price, st)
        if reason and not DRY_RUN:
            resp = place_order_with_log(sym, "sell", amt, restricted)
            if resp is not None:
                log.sell(sym, price, reason)
                sells += 1
                pos_state.pop(sym, None)
        elif reason and DRY_RUN:
            log.sell(sym, price, reason + " (DRY_RUN)")
            sells += 1
            pos_state.pop(sym, None)

    # Refresh balances after potential sells
    raw_bal = fetch_balance_raw()
    usd_cash = usd_cash_from_raw(raw_bal)

    # Count open positions (ignore dust)
    opens_after = 0
    for base, amt in raw_bal.items():
        bu=base.upper()
        if bu in ("USD","ZUSD"): continue
        sym = symbol_for_base(bu, markets)
        if not sym: continue
        _, ask = best_bid_ask(sym)
        if float(amt)*float(ask or 0) >= max(2.0, DUST_MIN_USD):
            opens_after += 1

    # --- BUY pass ---
    candidates = pick_candidates(usd_pairs, UNIVERSE_TOP_K)
    buys_allowed = max(0, MAX_BUYS_PER_RUN)

    for pair, last, ch in candidates:
        if buys >= buys_allowed: break
        if opens_after >= MAX_POSITIONS: break
        if is_restricted_symbol(pair, restricted):
            log.skipped(pair, "blacklisted (restricted)")
            continue

        _, ask = best_bid_ask(pair)
        px = ask or last
        if not px or px <= 0: continue
        if not can_afford_buy(usd_cash): break
        amount = float(f"{(MIN_BUY_USD/px):.8f}")
        if amount <= 0: continue

        if DRY_RUN:
            log.buy(pair, px)
            buys += 1; opens_after += 1; usd_cash -= MIN_BUY_USD
            pos_state[pair] = {"entry": px, "peak": px, "amount": amount}
        else:
            resp = place_order_with_log(pair, "buy", amount, restricted)
            if resp is None:
                # order rejected; if it was due to restriction, we've already blacklisted and logged
                continue
            log.buy(pair, px)
            buys += 1; opens_after += 1; usd_cash -= MIN_BUY_USD
            pos_state[pair] = {"entry": px, "peak": px, "amount": amount}

    # KPI
    try:
        bal_now = usd_cash_from_raw(fetch_balance_raw())
    except:
        bal_now = usd_cash
    try: append_kpi_csv(bal_now)
    except: pass

    # Save state/artifacts
    save_positions(pos_state)
    save_restricted(restricted)
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
