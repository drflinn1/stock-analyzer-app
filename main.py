#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crypto Live — Engine (Kraken, mean-revert buy + light rotation)

What it does (per run):
- Loads config from environment (wired by workflow).
- Authenticates to Kraken via ccxt (fails fast if secrets are missing).
- Fetches USD balance and a small USD-pair universe of markets.
- Ranks candidates by short-term dip (mean-revert scalp) and basic liquidity.
- Buys up to MAX_BUYS_PER_RUN small tickets (MIN_BUY_USD each),
  respecting MAX_POSITIONS and RESERVE_CASH_PCT.
- Optional light rotation when full/cash-short (sell weakest, buy stronger).
- Writes artifacts into .state/: run.log, kpi_history.csv (bal_usd), spec_gate_report.txt.
- Prints single-line BUY logs and a final SUMMARY line.

Environment (from workflow):
  DRY_RUN (ON/OFF)
  EXCHANGE = "kraken"
  KRAKEN_API_KEY / KRAKEN_API_SECRET
  MIN_BUY_USD
  MAX_POSITIONS
  MAX_BUYS_PER_RUN
  UNIVERSE_TOP_K
  RESERVE_CASH_PCT
  ROTATE_WHEN_FULL
  ROTATE_WHEN_CASH_SHORT
  DUST_MIN_USD
  DUST_SKIP_STABLES
"""

from __future__ import annotations
import os, sys, time, math, json, csv, pathlib, traceback, datetime as dt
from typing import Dict, List, Tuple
import ccxt
import pandas as pd

STATE_DIR = pathlib.Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG = STATE_DIR / "run.log"
KPI_CSV = STATE_DIR / "kpi_history.csv"
SPEC_TXT = STATE_DIR / "spec_gate_report.txt"

# ---------- tiny logger ----------
class Log:
    def __init__(self):
        self.lines: List[str] = []
    def _emit(self, level: str, msg: str):
        ts = dt.datetime.utcnow().isoformat() + "+00:00"
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self.lines.append(line)
    def info(self, msg: str): self._emit("INFO", msg)
    def warn(self, msg: str): self._emit("WARN", "warn: " + msg)
    def error(self, msg: str): self._emit("ERROR", "ERROR: " + msg)
    def buy(self, pair: str, price: float):
        self._emit("INFO", f"[BUY] {pair} @ {price:.5f}")
    def sell(self, pair: str, price: float):
        self._emit("INFO", f"[SELL] {pair} @ {price:.5f}")
    def summary(self, buys:int, sells:int, open_positions:int, dry_run:bool):
        self._emit("INFO", f"SUMMARY buys={buys} sells={sells} open={open_positions} DRY_RUN={dry_run}")

log = Log()

# ---------- env helpers ----------
def env_str(name: str, default: str="") -> str:
    v = os.getenv(name, default)
    return str(v)

def env_bool(name: str, default: str="false") -> bool:
    return env_str(name, default).strip().lower() in ("1","true","on","yes","y")

def env_float(name: str, default: str="0") -> float:
    try:
        return float(env_str(name, default))
    except Exception:
        return float(default)

def env_int(name: str, default: str="0") -> int:
    try:
        return int(float(env_str(name, default)))
    except Exception:
        return int(float(default))

# ---------- load config ----------
DRY_RUN       = env_str("DRY_RUN", "ON").upper() == "ON"
MIN_BUY_USD   = env_float("MIN_BUY_USD", "15")
MAX_POSITIONS = env_int("MAX_POSITIONS", "3")
MAX_BUYS_PER_RUN = env_int("MAX_BUYS_PER_RUN", "1")
UNIVERSE_TOP_K   = env_int("UNIVERSE_TOP_K", "25")
RESERVE_CASH_PCT = env_float("RESERVE_CASH_PCT", "5")
ROTATE_WHEN_FULL       = env_bool("ROTATE_WHEN_FULL", "true")
ROTATE_WHEN_CASH_SHORT = env_bool("ROTATE_WHEN_CASH_SHORT", "true")
DUST_MIN_USD    = env_float("DUST_MIN_USD", "2")
DUST_SKIP_STABLES = env_bool("DUST_SKIP_STABLES", "true")

API_KEY = env_str("KRAKEN_API_KEY", "")
API_SEC = env_str("KRAKEN_API_SECRET", "")

# ---------- ccxt exchange ----------
if not API_KEY or not API_SEC:
    raise RuntimeError("Kraken credentials missing: set KRAKEN_API_KEY and KRAKEN_API_SECRET in the workflow env.")

kraken = ccxt.kraken({
    "apiKey": API_KEY,
    "secret": API_SEC,
    "enableRateLimit": True,
    # kraken uses cost limits; use market by price/amount translation inside ccxt
})

# ---------- utility ----------
USD_STABLE_CODES = {"USDT","USDC","DAI","TUSD","FDUSD","USDP","GUSD","PYUSD"}

def write_artifacts():
    try:
        RUN_LOG.write_text("\n".join(log.lines), encoding="utf-8")
    except Exception:
        pass

def append_kpi_csv(bal_usd: float):
    KPI_CSV.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestamp_utc","bal_usd"]
    is_new = not KPI_CSV.exists()
    with KPI_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new: w.writerow(header)
        w.writerow([dt.datetime.utcnow().isoformat()+"+00:00", f"{bal_usd:.2f}"])

def write_spec(report: str):
    SPEC_TXT.write_text(report, encoding="utf-8")

def safe_get(dic, *keys, default=None):
    cur = dic
    for k in keys:
        if not isinstance(cur, dict) or k not in cur: return default
        cur = cur[k]
    return cur

# ---------- market + balance ----------
def fetch_usd_balance() -> float:
    try:
        bal = kraken.fetch_balance()
        # Kraken returns balances in 'total' with per-currency keys
        usd = 0.0
        for k,v in bal.get("total", {}).items():
            if k.upper() in ("USD","ZUSD"):
                usd += float(v or 0)
        return float(usd)
    except Exception as e:
        log.warn(f'fetch_balance failed: {e}')
        return 0.0

def list_usd_pairs(limit: int = 40) -> List[str]:
    markets = kraken.load_markets()
    pairs = []
    for s, m in markets.items():
        if not m.get("active", True): continue
        # choose cash market USD (not futures)
        q = m.get("quote", "").upper()
        if q == "USD":
            base = m.get("base", "").upper()
            if DUST_SKIP_STABLES and base in USD_STABLE_CODES:
                continue
            pairs.append(s)
    # keep a modest number
    return pairs[:max(limit,1)]

def pct_change_from_ohlcv(pair: str, timeframe="15m", candles=20) -> Tuple[float,float]:
    """
    Returns (last_close, pct_change_lookback) where pct_change_lookback is (last - prev)/prev.
    If data missing, returns (nan, -inf) to de-prioritize.
    """
    try:
        ohlcv = kraken.fetch_ohlcv(pair, timeframe=timeframe, limit=candles)
        if not ohlcv or len(ohlcv) < 3:
            return (math.nan, float("-inf"))
        last = ohlcv[-1][4]
        prev = ohlcv[-3][4]   # look back 2 candles for stability
        if prev and prev > 0:
            return (float(last), (last - prev) / prev)
        return (float(last), float("-inf"))
    except Exception:
        return (math.nan, float("-inf"))

def best_bid_ask(pair: str) -> Tuple[float,float]:
    try:
        ob = kraken.fetch_ticker(pair)
        ask = safe_get(ob, "ask", default=None)
        bid = safe_get(ob, "bid", default=None)
        if not ask or ask <= 0:
            ask = safe_get(ob, "last", default=None)
        return float(bid or 0), float(ask or 0)
    except Exception:
        return (0.0, 0.0)

def portfolio_positions(bal: Dict[str,float]) -> List[Tuple[str,float]]:
    """Return list of (currency, amount) excluding (USD/ZUSD) and tiny dust."""
    out = []
    for c, amt in bal.items():
        cu = c.upper()
        if cu in ("USD","ZUSD"): continue
        try:
            a = float(amt or 0)
        except Exception:
            a = 0.0
        if a <= 0: continue
        out.append((cu, a))
    return out

def fetch_balances_raw() -> Dict[str, float]:
    try:
        b = kraken.fetch_balance()
        return {k: float(v or 0) for k,v in b.get("total", {}).items()}
    except Exception as e:
        log.warn(f'fetch_balance failed: {e}')
        return {}

# ---------- trade ops ----------
def place_order_with_log(params: dict) -> dict:
    try:
        resp = kraken.create_order(**params)
    except Exception as e:
        log.error(f'EXCHANGE_ERROR AddOrder exception: {e} | params={{"symbol":"{params.get("symbol")}","type":"{params.get("type")}","side":"{params.get("side")}","amount":{params.get("amount")},"price":{params.get("price")}}}')
        raise
    # ccxt throws on exchange-level errors; if here, assume accepted
    txid = safe_get(resp, "id", default=None)
    log.info(f'EXCHANGE OK AddOrder accepted | txid={txid}')
    return resp

# ---------- strategy ----------
def pick_candidates(pairs: List[str], top_k: int) -> List[Tuple[str,float,float]]:
    """
    Rank by most negative short-term % change (dip) to mean-revert.
    Returns list of (pair, last_price, pct_change) sorted asc by pct_change.
    """
    rows = []
    for p in pairs:
        last, ch = pct_change_from_ohlcv(p)
        if not last or not math.isfinite(ch):
            continue
        rows.append((p, last, ch))
    rows.sort(key=lambda r: r[2])  # most negative first
    return rows[:max(1, top_k)]

def can_afford_buy(usd_free: float) -> bool:
    reserve = RESERVE_CASH_PCT / 100.0
    kept = usd_free * (1 - reserve)  # we can spend up to (1-reserve)*cash
    return kept >= MIN_BUY_USD + 0.01

def symbol_for_base(base: str, markets: Dict[str, dict]) -> str | None:
    # find BASE/USD market symbol
    for s, m in markets.items():
        if m.get("base","").upper() == base.upper() and m.get("quote","").upper() == "USD" and m.get("active",True):
            return s
    return None

def main():
    start = time.time()
    buys = 0
    sells = 0

    # Spec/gate report for artifacts
    spec_lines = []
    spec_lines.append(f"DRY_RUN={DRY_RUN}")
    spec_lines.append(f"MIN_BUY_USD={MIN_BUY_USD}")
    spec_lines.append(f"MAX_POSITIONS={MAX_POSITIONS} MAX_BUYS_PER_RUN={MAX_BUYS_PER_RUN}")
    spec_lines.append(f"UNIVERSE_TOP_K={UNIVERSE_TOP_K} RESERVE_CASH_PCT={RESERVE_CASH_PCT}")
    spec_lines.append(f"ROTATE_WHEN_FULL={ROTATE_WHEN_FULL} ROTATE_WHEN_CASH_SHORT={ROTATE_WHEN_CASH_SHORT}")
    spec_lines.append(f"DUST_MIN_USD={DUST_MIN_USD} DUST_SKIP_STABLES={DUST_SKIP_STABLES}")

    # Load markets & balances
    markets = kraken.load_markets()
    usd_pairs = list_usd_pairs(limit=80)

    raw_bal = fetch_balances_raw()
    usd_cash = 0.0
    for k,v in raw_bal.items():
        if k.upper() in ("USD","ZUSD"):
            usd_cash += float(v)

    # current open positions count (roughly any non-USD with > dust)
    opens = 0
    for c,a in portfolio_positions(raw_bal):
        # estimate USD value if we can map base->symbol
        sym = symbol_for_base(c, markets)
        if not sym: continue
        _, ask = best_bid_ask(sym)
        value = a * (ask or 0)
        if value >= max(2.0, DUST_MIN_USD):  # ignore dust
            opens += 1

    # Candidate selection
    candidates = pick_candidates(usd_pairs, UNIVERSE_TOP_K)

    # Buy loop
    buys_allowed = max(0, MAX_BUYS_PER_RUN)
    # optionally rotate if full or cash short
    rotate = False
    if opens >= MAX_POSITIONS and ROTATE_WHEN_FULL:
        rotate = True
        spec_lines.append("Rotation reason: FULL")
    if not can_afford_buy(usd_cash) and ROTATE_WHEN_CASH_SHORT:
        rotate = True
        spec_lines.append("Rotation reason: CASH_SHORT")

    # sell-light rotation (optional): sell weakest open to free cash
    if rotate and opens > 0 and not DRY_RUN:
        # find weakest holding by last 15m change
        weakest = None
        weak_change = 999.0
        for c,a in portfolio_positions(raw_bal):
            sym = symbol_for_base(c, markets)
            if not sym: continue
            last, ch = pct_change_from_ohlcv(sym)
            if not math.isfinite(ch): continue
            if ch < weak_change:
                weakest, weak_change = (sym, (last or 0)), ch
        if weakest:
            sym, last_price = weakest
            # market sell all (best effort)
            try:
                pos_amt = 0.0
                # raw_bal contains base amounts, need amount in base units
                base = markets[sym]["base"]
                pos_amt = raw_bal.get(base, 0.0)
                if pos_amt > 0:
                    params = {"symbol": sym, "type":"market", "side":"sell", "amount": pos_amt}
                    place_order_with_log(params)
                    log.sell(sym, last_price or 0.0)
                    sells += 1
                    # refresh balances
                    raw_bal = fetch_balances_raw()
                    usd_cash = float(raw_bal.get("USD",0.0) + raw_bal.get("ZUSD",0.0))
            except Exception as e:
                log.warn(f"rotation sell failed {sym}: {e}")

    # recalc opens after rotation
    opens_after = opens

    for pair, last, ch in candidates:
        if buys >= buys_allowed:
            break
        # capacity
        if opens_after >= MAX_POSITIONS:
            break

        # price / sizing
        bid, ask = best_bid_ask(pair)
        px = ask or last
        if not px or px <= 0:
            continue

        # spend gate
        if not can_afford_buy(usd_cash):
            break

        # amount sized to spend ~ MIN_BUY_USD
        amount = max(0.0, MIN_BUY_USD / px)
        # Kraken min order amounts vary; ccxt should raise if too small.
        # We'll add a small epsilon for rounding.
        amount = float(f"{amount:.8f}")

        if amount <= 0:
            continue

        if DRY_RUN:
            log.buy(pair, px)
            buys += 1
            opens_after += 1
            usd_cash -= MIN_BUY_USD
        else:
            try:
                params = {"symbol": pair, "type":"market", "side":"buy", "amount": amount}
                place_order_with_log(params)
                log.buy(pair, px)
                buys += 1
                opens_after += 1
                usd_cash -= MIN_BUY_USD
            except Exception as e:
                log.warn(f"buy failed {pair}: {e}")

    # KPI balance snapshot
    # Try again to read fresh USD after trades; fall back to previous.
    bal_usd_now = fetch_usd_balance() or usd_cash
    try:
        append_kpi_csv(bal_usd_now)
    except Exception:
        pass

    # finish
    log.summary(buys=buys, sells=sells, open_positions=opens_after, dry_run=DRY_RUN)
    write_artifacts()
    write_spec("\n".join(spec_lines))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
            # Print a final error but still write artifacts so we can inspect.
            log.error(f"fatal: {e}\n{traceback.format_exc()}")
            write_artifacts()
            raise
