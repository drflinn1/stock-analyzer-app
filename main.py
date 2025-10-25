#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crypto Live — Engine (Kraken)

Features
- Mean-revert buys (dip ranking on 15m)
- Exits: TAKE_PROFIT, STOP_LOSS, TRAILING (armed)
- Restricted-market auto-blacklist (jurisdiction errors)
- KPI + artifacts in .state/
- Profit Flags   -> run.log + .state/profit_flags.txt
- Artifact hardening (always creates .state files)
- Entry recovery from trade history if .state is empty
"""

from __future__ import annotations
import os, math, json, csv, pathlib, traceback, datetime as dt
from typing import Dict, List, Tuple
import ccxt

# ---------------- paths ----------------
STATE_DIR = pathlib.Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG  = STATE_DIR / "run.log"
KPI_CSV  = STATE_DIR / "kpi_history.csv"
SPEC_TXT = STATE_DIR / "spec_gate_report.txt"
POS_JSON = STATE_DIR / "spot_positions.json"
RESTRICTED_JSON = STATE_DIR / "restricted_markets.json"
PROFIT_FLAGS_TXT = STATE_DIR / "profit_flags.txt"

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
    def profit_flag(self,pair,gain,band): self._emit(f"PROFIT_FLAG {pair} +{gain:.2f}% ≥ {band}%")
    def summary(self,buys,sells,open_,dry): self._emit(f"SUMMARY buys={buys} sells={sells} open={open_} DRY_RUN={dry}")

log = Log()

# ---------------- env helpers ----------------
def env_str(n,d=""): return str(os.getenv(n, d))
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
STOP_LOSS_PCT      = env_float("STOP_LOSS_PCT","5.0")       # STOP_LOSS keyword
TRAILING_STOP_PCT  = env_float("TRAILING_STOP_PCT","2.0")   # trailing keyword
TRAILING_ARM_PCT   = env_float("TRAILING_ARM_PCT","1.5")

# Restricted toggle
SKIP_RESTRICTED = env_bool("SKIP_RESTRICTED","true")

API_KEY = env_str("KRAKEN_API_KEY","")
API_SEC = env_str("KRAKEN_API_SECRET","")
if not API_KEY or not API_SEC:
    raise RuntimeError("Kraken credentials missing: set KRAKEN_API_KEY and KRAKEN_API_SECRET.")

kraken = ccxt.kraken({"apiKey":API_KEY, "secret":API_SEC, "enableRateLimit":True})

USD_STABLE_CODES = {"USDT","USDC","DAI","TUSD","FDUSD","USDP","GUSD","PYUSD"}

# ---------------- file helpers ----------------
def write_text(path: pathlib.Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def ensure_artifact_shells():
    # Create placeholder files so artifact step always finds something
    for p in (RUN_LOG, KPI_CSV, SPEC_TXT, PROFIT_FLAGS_TXT):
        if not p.exists():
            if p.suffix.lower()==".csv":
                p.write_text("timestamp_utc,bal_usd\n", encoding="utf-8")
            else:
                p.write_text("", encoding="utf-8")

def write_artifacts(lines: List[str], spec: str):
    ensure_artifact_shells()
    try: RUN_LOG.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    except: pass
    try: SPEC_TXT.write_text(spec, encoding="utf-8")
    except: pass

def append_kpi_csv(bal_usd: float):
    ensure_artifact_shells()
    if not KPI_CSV.exists():
        KPI_CSV.write_text("timestamp_utc,bal_usd\n", encoding="utf-8")
    with KPI_CSV.open("a", newline="", encoding="utf-8") as f:
        f.write(f"{dt.datetime.utcnow().isoformat()}+00:00,{bal_usd:.2f}\n")

def load_json(path: pathlib.Path, default):
    if path.exists():
        try: return json.loads(path.read_text())
        except: return default
    return default

def save_json(path: pathlib.Path, obj):
    path.write_text(json.dumps(obj, indent=2))

# ---------------- exchange helpers ----------------
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
        bid = t.get("bid") or t.get("last")
        ask = t.get("ask") or t.get("last")
        return float(bid or 0), float(ask or 0)
    except: return (0.0,0.0)

def place_order_with_log(symbol:str, side:str, amount:float, restricted: Dict[str,bool]):
    try:
        resp = kraken.create_order(symbol=symbol, type="market", side=side, amount=amount)
        txid = resp.get("id")
        log.info(f'EXCHANGE OK AddOrder accepted | txid={txid} | side={side} symbol={symbol} amount={amount}')
        return resp
    except Exception as e:
        emsg = str(e)
        if "Invalid permissions" in emsg or "restricted for" in emsg or "EAccount" in emsg:
            # auto-blacklist
            if not restricted.get(symbol):
                restricted[symbol]=True
                save_json(RESTRICTED_JSON, restricted)
            log.skipped(symbol, emsg)
            return None
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

# ---------------- SELL rules ----------------
def check_sell_rules(price:float, entry:float, peak:float) -> str|None:
    if entry<=0 or price<=0: return None
    gain = (price/entry - 1.0) * 100.0
    dd_from_peak = (1.0 - price/max(peak, entry)) * 100.0
    if gain >= TAKE_PROFIT_PCT:
        return f"TAKE_PROFIT +{gain:.2f}% ≥ {TAKE_PROFIT_PCT:.2f}%"
    if gain <= -abs(STOP_LOSS_PCT):
        return f"STOP_LOSS {gain:.2f}% ≤ -{abs(STOP_LOSS_PCT):.2f}%"
    if gain >= TRAILING_ARM_PCT and dd_from_peak >= TRAILING_STOP_PCT:
        return f"TRAILING_DROP {dd_from_peak:.2f}% ≥ {TRAILING_STOP_PCT:.2f}% (armed at +{TRAILING_ARM_PCT:.2f}%)"
    return None

# ---------------- Profit Flags ----------------
GAIN_BANDS = [10, 25, 50, 100, 200]  # % thresholds to announce

def emit_profit_flags(open_rows: List[Tuple[str,float,float,float]]):
    """open_rows: (symbol, amount, price, entry)"""
    lines = []
    now = dt.datetime.utcnow().isoformat()+"+00:00"
    lines.append(f"# Profit Flags {now}")
    ranked = []

    for sym, amt, price, entry in open_rows:
        if entry <= 0 or price <= 0: continue
        gain = (price/entry - 1.0) * 100.0
        ranked.append((gain, sym, price, entry, amt))
        for band in reversed(GAIN_BANDS):
            if gain >= band:
                log.profit_flag(sym, gain, band)
                break

    ranked.sort(reverse=True)  # highest gain first
    for gain, sym, price, entry, amt in ranked:
        lines.append(f"{sym}  gain=+{gain:.2f}%  price={price:.6f}  entry={entry:.6f}  amt={amt}")

    write_text(PROFIT_FLAGS_TXT, "\n".join(lines)+"\n")

# ---------------- Entry recovery from real trades ----------------
def seed_entry_from_trades(symbol: str) -> float:
    """
    Compute a VWAP entry for the current net long position using recent trades.
    Returns 0.0 if trades cannot be fetched or there is no net long qty.
    """
    try:
        trades = kraken.fetch_my_trades(symbol, limit=200)  # recent window is enough for spot
    except Exception:
        return 0.0

    net_qty = 0.0
    gross_cost = 0.0
    for t in trades:
        amt = float(t.get("amount") or 0)           # +buy, -sell
        price = float(t.get("price") or 0)
        fee = float((t.get("fee") or {}).get("cost") or 0)
        if amt > 0:
            net_qty += amt
            gross_cost += amt * price + fee
        elif amt < 0:
            take = min(abs(amt), max(0.0, net_qty))
            if net_qty > 0 and gross_cost > 0 and take > 0:
                avg = gross_cost / net_qty
                gross_cost -= avg * take
            net_qty += amt  # amt is negative

    if net_qty > 0 and gross_cost > 0:
        return gross_cost / net_qty
    return 0.0

# ---------------- main ----------------
def main():
    ensure_artifact_shells()

    buys = sells = 0
    spec = [
        f"DRY_RUN={DRY_RUN}",
        f"MIN_BUY_USD={MIN_BUY_USD}",
        f"MAX_POSITIONS={MAX_POSITIONS} MAX_BUYS_PER_RUN={MAX_BUYS_PER_RUN}",
        f"UNIVERSE_TOP_K={UNIVERSE_TOP_K} RESERVE_CASH_PCT={RESERVE_CASH_PCT}",
        f"ROTATE_WHEN_FULL={ROTATE_WHEN_FULL} ROTATE_WHEN_CASH_SHORT={ROTATE_WHEN_CASH_SHORT}",
        f"DUST_MIN_USD={DUST_MIN_USD} DUST_SKIP_STABLES={DUST_SKIP_STABLES}",
        f"TAKE_PROFIT_PCT={TAKE_PROFIT_PCT} STOP_LOSS_PCT={STOP_LOSS_PCT} "
        f"TRAILING_STOP_PCT={TRAILING_STOP_PCT} TRAILING_ARM_PCT={TRAILING_ARM_PCT}",
        f"SKIP_RESTRICTED={SKIP_RESTRICTED}",
    ]

    markets = kraken.load_markets()
    usd_pairs = list_usd_pairs(limit=80)
    restricted = load_json(RESTRICTED_JSON, {})

    if SKIP_RESTRICTED:
        usd_pairs = [s for s in usd_pairs if not restricted.get(s, False)]

    # balances
    raw_bal = fetch_balance_raw()
    usd_cash = usd_cash_from_raw(raw_bal)

    # positions state
    pos_state = load_json(POS_JSON, {})

    # ---- SELL pass over open holdings ----
    open_rows_for_flags = []  # (symbol, amt, price, entry)
    opens_after = 0

    for base, amt in raw_bal.items():
        bu = str(base).upper()
        if bu in ("USD","ZUSD"): continue
        sym = symbol_for_base(bu, markets)
        if not sym: continue
        if SKIP_RESTRICTED and restricted.get(sym, False):
            log.skipped(sym, "present in holdings but restricted for jurisdiction")
            continue

        _, ask = best_bid_ask(sym)
        value = float(amt) * float(ask or 0)
        if value < max(2.0, DUST_MIN_USD):
            continue
        opens_after += 1

        # ---- init / update state with trade-based entry recovery ----
        st = pos_state.get(sym, {})
        entry = float(st.get("entry") or 0.0)
        peak  = float(st.get("peak")  or 0.0)

        if entry <= 0.0:
            recovered = seed_entry_from_trades(sym)
            entry = float(recovered or (ask or 0.0))
            if recovered > 0:
                log.info(f"ENTRY_RECOVERED {sym} entry={entry:.6f} (from trades)")
        if peak <= 0.0:
            peak = float(ask or entry or 0.0)

        if (ask or 0.0) > peak:
            peak = float(ask or 0.0)

        pos_state[sym] = {"entry": float(entry), "peak": float(peak), "amount": float(amt)}

        # ---- sell rules
        reason = check_sell_rules(float(ask or 0), float(entry), float(peak))
        gain_pct = (float(ask or 0)/entry - 1.0) * 100.0 if entry > 0 else float('nan')
        dd_pct = (1.0 - float(ask or 0)/max(peak, entry)) * 100.0 if entry > 0 else float('nan')
        log.info(f"SELL_CHECK {sym} price={float(ask or 0):.6f} entry={entry:.6f} peak={peak:.6f} gain={gain_pct:.2f}% dd_from_peak={dd_pct:.2f}%")

        if reason and not DRY_RUN:
            resp = place_order_with_log(sym, "sell", float(amt), restricted)
            if resp is not None:
                log.sell(sym, float(ask or 0), reason)
                sells += 1
                pos_state.pop(sym, None)
                opens_after -= 1
        elif reason and DRY_RUN:
            log.sell(sym, float(ask or 0), reason + " (DRY_RUN)")
            sells += 1
            pos_state.pop(sym, None)
            opens_after -= 1
        else:
            open_rows_for_flags.append((sym, float(amt), float(ask or 0), float(entry)))

    # refresh cash after sells
    raw_bal = fetch_balance_raw()
    usd_cash = usd_cash_from_raw(raw_bal)

    # ---- BUY pass (mean-revert) ----
    candidates = pick_candidates(usd_pairs, UNIVERSE_TOP_K)
    buys_allowed = max(0, MAX_BUYS_PER_RUN)

    for pair, last, ch in candidates:
        if buys >= buys_allowed: break
        if opens_after >= MAX_POSITIONS: break
        if SKIP_RESTRICTED and restricted.get(pair, False):
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
            pos_state[pair] = {"entry": float(px), "peak": float(px), "amount": float(amount)}
            open_rows_for_flags.append((pair, float(amount), float(px), float(px)))
        else:
            resp = place_order_with_log(pair, "buy", amount, restricted)
            if resp is None:
                continue
            log.buy(pair, px)
            buys += 1; opens_after += 1; usd_cash -= MIN_BUY_USD
            pos_state[pair] = {"entry": float(px), "peak": float(px), "amount": float(amount)}
            open_rows_for_flags.append((pair, float(amount), float(px), float(px)))

    # ---- Profit Flags & KPI ----
    try: emit_profit_flags(open_rows_for_flags)
    except Exception as e: log.warn(f"profit-flag emit failed: {e}")

    try:
        bal_now = usd_cash_from_raw(fetch_balance_raw())
    except Exception:
        bal_now = usd_cash
    try: append_kpi_csv(bal_now)
    except Exception: pass

    # ---- Save state & artifacts ----
    save_json(POS_JSON, pos_state)
    save_json(RESTRICTED_JSON, restricted)

    log.summary(buys, sells, opens_after, DRY_RUN)
    write_artifacts(log.lines, "\n".join(spec))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"fatal: {e}\n{traceback.format_exc()}")
        write_artifacts(log.lines, "fatal error")
        raise
