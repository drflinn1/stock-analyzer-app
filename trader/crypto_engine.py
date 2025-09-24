# trader/crypto_engine.py
# Auto-pick USD pairs on Kraken, trend filter (1h EMA12>EMA26 & RSI>50),
# per-symbol cooldown, TP/SL exits, USD/ZUSD balance, DRY_RUN safe.
# Handles exchange min cost/amount & precision; keeps simple state in .state/.

from __future__ import annotations
import os, json, time, math, glob
from typing import Dict, Any, List, Tuple, Optional
from decimal import Decimal, ROUND_DOWN

import ccxt
import pandas as pd
import numpy as np

# ---------- env helpers ----------
def envs(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default

def envb(name: str, default: bool) -> bool:
    return envs(name, str(default)).strip().lower() in ("1","true","yes","y","on")

def envf(name: str, default: float) -> float:
    try: return float(envs(name, str(default)))
    except: return default

def envi(name: str, default: int) -> int:
    try: return int(float(envs(name, str(default))))
    except: return default

def log(msg: str) -> None:
    print(msg, flush=True)

# ---------- config (tweaks & autopick) ----------
DRY_RUN          = envb("DRY_RUN", False)
EXCHANGE_ID      = envs("EXCHANGE_ID", "kraken")
API_KEY          = envs("API_KEY", "")
API_SECRET       = envs("API_SECRET", "")
API_PASSWORD     = envs("API_PASSWORD", "")

PER_TRADE_USD    = envf("PER_TRADE_USD", 10.0)   # tweaked
DAILY_CAP_USD    = envf("DAILY_CAP_USD", 20.0)   # tweaked
AVOID_REBUY      = envb("AVOID_REBUY", True)
TP_PCT           = envf("TAKE_PROFIT_PCT", 0.025)  # tweaked
SL_PCT           = envf("STOP_LOSS_PCT", 0.015)     # tweaked

AUTO_PICK        = envb("AUTO_PICK", True)
QUOTE_CCY        = envs("QUOTE_CCY", "USD").upper()
PICKS_PER_RUN    = envi("PICKS_PER_RUN", 2)
MIN_QUOTE_VOL_USD= envf("MIN_QUOTE_VOL_USD", 2_000_000)  # 24h quote$
TREND_FILTER     = envb("TREND_FILTER", True)
COOLDOWN_MINUTES = envi("COOLDOWN_MINUTES", 60)

# If AUTO_PICK=false, you can still set UNIVERSE manually (comma list)
UNIVERSE_MANUAL  = [s.strip().upper() for s in envs("UNIVERSE", "").split(",") if s.strip()]

STATE_DIR   = ".state"
STATE_FILE  = os.path.join(STATE_DIR, "crypto_positions.json")
COOLDOWN_FILE = os.path.join(STATE_DIR, "crypto_cooldown.json")
USD_KEYS    = ("USD","ZUSD")

# ---------- utils ----------
def q_round(x: float, decimals: int) -> float:
    q = Decimal(str(x)).quantize(Decimal("1e-%d" % decimals), rounding=ROUND_DOWN)
    return float(q)

def build_exchange() -> ccxt.Exchange:
    klass = getattr(ccxt, EXCHANGE_ID)
    params = {"enableRateLimit": True, "options": {"warnOnFetchOpenOrdersWithoutSymbol": False}}
    if API_PASSWORD:
        params["password"] = API_PASSWORD
    ex = klass({"apiKey": API_KEY, "secret": API_SECRET, **params})
    ex.load_markets()
    return ex

def ensure_state() -> Dict[str, Any]:
    os.makedirs(STATE_DIR, exist_ok=True)
    if not os.path.isfile(STATE_FILE):
        data = {"entries":{}, "spent_today":0.0, "day": pd.Timestamp.utcnow().date().isoformat()}
        with open(STATE_FILE, "w") as f: json.dump(data, f)
    else:
        with open(STATE_FILE, "r") as f: data = json.load(f)
    today = pd.Timestamp.utcnow().date().isoformat()
    if data.get("day") != today:
        data["day"] = today; data["spent_today"] = 0.0
    data.setdefault("entries",{})
    return data

def load_cooldown() -> Dict[str,int]:
    if not os.path.isfile(COOLDOWN_FILE): return {}
    try:
        with open(COOLDOWN_FILE, "r") as f: return json.load(f)
    except: return {}

def save_cooldown(d: Dict[str,int]) -> None:
    with open(COOLDOWN_FILE, "w") as f: json.dump(d, f, indent=2, sort_keys=True)

def save_state(d: Dict[str,Any]) -> None:
    with open(STATE_FILE, "w") as f: json.dump(d, f, indent=2, sort_keys=True)

def usd_balance(b: Dict[str, Any]) -> float:
    for k in USD_KEYS:
        if k in b and isinstance(b[k], dict):
            v = b[k].get("free", b[k].get("total", 0.0))
            try: return float(v)
            except: pass
    return 0.0

def base_from_symbol(symbol: str) -> str:
    return symbol.split("/")[0]

# ---------- market info ----------
def fetch_price(ex: ccxt.Exchange, symbol: str) -> Optional[float]:
    try:
        t = ex.fetch_ticker(symbol); p = t.get("last") or t.get("close")
        return float(p) if p else None
    except Exception as e:
        log(f"[PRICE] {symbol} fail: {e}"); return None

def market_limits(ex: ccxt.Exchange, symbol: str) -> Tuple[float,float,int,int]:
    m = ex.market(symbol)
    limits = m.get("limits", {})
    min_cost = float(limits.get("cost", {}).get("min") or 0.0)
    min_amount = float(limits.get("amount", {}).get("min") or 0.0)
    pp = int(m.get("precision", {}).get("price", 8))
    ap = int(m.get("precision", {}).get("amount", 8))
    return min_cost, min_amount, pp, ap

# ---------- trend filter (1h) ----------
def ema(s: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(s).ewm(span=span, adjust=False).mean().values

def rsi(close: np.ndarray, period: int = 14) -> float:
    delta = np.diff(close)
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    ru = pd.Series(up).ewm(alpha=1/period, adjust=False).mean()
    rd = pd.Series(down).ewm(alpha=1/period, adjust=False).mean()
    rs = ru / (rd + 1e-9)
    return float(100 - (100 / (1 + rs.iloc[-1])))

def trend_ok(ex: ccxt.Exchange, symbol: str) -> bool:
    if not TREND_FILTER: return True
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe="1h", limit=90)
        if not ohlcv or len(ohlcv) < 30: return True
        close = np.array([c[4] for c in ohlcv], dtype=float)
        e12 = ema(close, 12)[-1]
        e26 = ema(close, 26)[-1]
        r = rsi(close, 14)
        good = (e12 > e26) and (r > 50)
        log(f"[TREND] {symbol} EMA12={e12:.4f} EMA26={e26:.4f} RSI={r:.1f} -> {good}")
        return good
    except Exception as e:
        log(f"[TREND] {symbol} fetch fail: {e}"); return True  # don't block on data hiccups

# ---------- auto-pick ----------
def autopick_symbols(ex: ccxt.Exchange, quote: str, picks: int, min_quote_usd: float) -> List[str]:
    # Build list of spot symbols with given quote; rank by 24h quote vol (fallback: last*baseVolume)
    out = []
    try:
        tickers = ex.fetch_tickers()
        for sym, t in tickers.items():
            try:
                m = ex.market(sym)
                if not m.get("spot"): continue
                if m.get("quote") != quote: continue
                qv = t.get("quoteVolume")
                if qv is None:
                    last = t.get("last") or t.get("close")
                    bv = t.get("baseVolume")
                    qv = (last or 0.0) * (bv or 0.0)
                if qv and qv >= min_quote_usd:
                    out.append((sym, float(qv)))
            except Exception:
                continue
        out.sort(key=lambda x: x[1], reverse=True)
        picks_syms = [s for s, _ in out[: max(5, picks*3)]]
        log(f"[AUTO] top by quoteVol≥{min_quote_usd:,.0f} ({quote}): {picks_syms[:10]}")
        # Apply trend filter and cut to requested count
        final = []
        for s in picks_syms:
            if len(final) >= picks: break
            if trend_ok(ex, s):
                final.append(s)
        log(f"[AUTO] picks: {final}")
        return final
    except Exception as e:
        log(f"[AUTO] fetch_tickers fail: {e}")
        return []

# ---------- balances & orders ----------
def current_base_balance(ex: ccxt.Exchange, base: str) -> float:
    try:
        bal = ex.fetch_balance()
        if base in bal and isinstance(bal[base], dict):
            v = bal[base].get("free", bal[base].get("total", 0.0))
            return float(v)
        return 0.0
    except Exception as e:
        log(f"[BAL] base {base} fail: {e}"); return 0.0

def place_market_sell_all(ex: ccxt.Exchange, symbol: str, qty: float) -> bool:
    if qty <= 0: return False
    log(f"[SELL] {symbol} qty={qty}")
    if DRY_RUN: return True
    try:
        ex.create_order(symbol=symbol, type="market", side="sell", amount=qty)
        log(f"[SELL-OK] {symbol}"); return True
    except Exception as e:
        log(f"[SELL-FAIL] {symbol}: {e}"); return False

def place_market_buy(ex: ccxt.Exchange, symbol: str, notional_usd: float, remaining: float) -> Optional[Tuple[float, float, float]]:
    price = fetch_price(ex, symbol)
    if not price or price <= 0:
        log(f"[BUY] {symbol} skip (no price)"); return None

    min_cost, min_amount, pp, ap = market_limits(ex, symbol)

    # raise notional to min cost if needed
    need = max(notional_usd, min_cost or 0.0)
    qty = need / price

    # if min amount binds, raise qty & notional accordingly
    if min_amount and qty < min_amount:
        qty = min_amount
        need = qty * price

    if need > remaining + 1e-9:
        log(f"[BUY] {symbol} need ${need:.2f} (min), remaining ${remaining:.2f} -> skip")
        return None

    qty = max(min_amount or 0.0, q_round(qty, ap))
    if qty <= 0:
        log(f"[BUY] {symbol} qty<=0 after precision -> skip"); return None

    log(f"[BUY] {symbol} qty={qty} price≈{q_round(price, pp)} notional≈${qty*price:.2f} (min_cost={min_cost or 0}, min_amount={min_amount or 0})")

    if DRY_RUN:
        return (qty, price, need)

    try:
        order = ex.create_order(symbol=symbol, type="market", side="buy", amount=qty)
        filled = float(order.get("filled") or qty)
        avg = float(order.get("average") or price)
        log(f"[BUY-OK] {symbol} filled={filled} avg={avg}")
        return (filled, avg, need)
    except Exception as e:
        log(f"[BUY-FAIL] {symbol}: {e}")
        return None

# ---------- main ----------
def main() -> None:
    log("=== Crypto Engine Start ===")
    log(f"DRY_RUN={DRY_RUN} EXCHANGE={EXCHANGE_ID} PER_TRADE_USD={PER_TRADE_USD} DAILY_CAP_USD={DAILY_CAP_USD}")
    log(f"AUTO_PICK={AUTO_PICK} QUOTE={QUOTE_CCY} PICKS={PICKS_PER_RUN} MIN_QUOTE_VOL_USD={MIN_QUOTE_VOL_USD}")
    log(f"TP={TP_PCT} SL={SL_PCT} AVOID_REBUY={AVOID_REBUY} TREND_FILTER={TREND_FILTER} COOL={COOLDOWN_MINUTES}m")

    ex = build_exchange()
    state = ensure_state()
    cooldown = load_cooldown()
    now = int(time.time())

    # Balances
    try: bal = ex.fetch_balance()
    except Exception as e: log(f"[BAL] fetch_balance failed: {e}"); bal = {}
    usd = usd_balance(bal)
    log(f"USD balance (USD/ZUSD): ${usd:,.2f}")

    # TP/SL sweep
    for symbol, entry in list(state["entries"].items()):
        price = fetch_price(ex, symbol)
        if not price: continue
        entry_px = float(entry["entry_price"])
        base = base_from_symbol(symbol)
        qty = current_base_balance(ex, base)
        if qty <= 0:
            log(f"[POS] {symbol} gone; rm"); state["entries"].pop(symbol, None); continue
        up = (price / entry_px) - 1.0
        down = 1.0 - (price / entry_px)
        if up >= TP_PCT:
            log(f"[TP] {symbol} p={price:.8f} e={entry_px:.8f} +{up:.2%}")
            if place_market_sell_all(ex, symbol, qty): state["entries"].pop(symbol, None); continue
        if down >= SL_PCT:
            log(f"[SL] {symbol} p={price:.8f} e={entry_px:.8f} -{down:.2%}")
            if place_market_sell_all(ex, symbol, qty): state["entries"].pop(symbol, None); continue

    save_state(state)

    # Budget check
    remaining = max(0.0, DAILY_CAP_USD - float(state.get("spent_today", 0.0)))
    log(f"Daily remaining budget: ${remaining:.2f}")
    if remaining < 0.01:
        log("No budget."); return

    # Build universe
    if AUTO_PICK:
        symbols = autopick_symbols(ex, QUOTE_CCY, PICKS_PER_RUN, MIN_QUOTE_VOL_USD)
    else:
        symbols = UNIVERSE_MANUAL

    # Place buys with cooldown & AVOID_REBUY
    for symbol in symbols:
        if remaining < 0.01: break

        # cooldown gate
        next_ok = int(cooldown.get(symbol, 0))
        if now < next_ok:
            left = next_ok - now
            log(f"[COOL] {symbol} wait {left//60}m"); continue

        # re-buy guards
        if AVOID_REBUY and symbol in state["entries"]:
            log(f"[SKIP] {symbol} (state held)"); continue
        base = base_from_symbol(symbol)
        if AVOID_REBUY and current_base_balance(ex, base) > 0:
            log(f"[SKIP] {symbol} (base balance>0)"); continue

        res = place_market_buy(ex, symbol, PER_TRADE_USD, remaining)
        if not res: continue

        filled_qty, avg_px, need = res
        state["entries"][symbol] = {"entry_price": float(avg_px), "ts": int(time.time())}
        state["spent_today"] = float(state.get("spent_today", 0.0)) + float(need)
        remaining = max(0.0, DAILY_CAP_USD - state["spent_today"])
        save_state(state)

        cooldown[symbol] = int(time.time()) + COOLDOWN_MINUTES * 60
        save_cooldown(cooldown)

    log("=== Crypto Engine Done ===")

if __name__ == "__main__":
    main()
