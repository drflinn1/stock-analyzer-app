# trader/crypto_engine.py
# Minimal, sturdy spot-crypto engine (Kraken-friendly) with:
# - USD/ZUSD balance detection
# - AVOID_REBUY guard
# - Market buys sized by USD notional
# - Simple TP/SL based on remembered entry price (state file)
# - DRY_RUN support and safe logging
#
# This is intentionally conservative and exchange-agnostic.
# It won't place OCO; it fires market exits when TP/SL conditions are met.

from __future__ import annotations
import os, json, time, math
from typing import Dict, Any, List, Tuple, Optional

import ccxt
import pandas as pd

# ---------- env helpers ----------
def getenv_str(k: str, d: str) -> str:
    v = os.environ.get(k)
    return v if v not in (None, "") else d

def getenv_bool(k: str, d: bool) -> bool:
    v = getenv_str(k, str(d)).strip().lower()
    return v in ("1","true","yes","y","on")

def getenv_float(k: str, d: float) -> float:
    try: return float(getenv_str(k, str(d)))
    except: return d

def log(msg: str) -> None:
    print(msg, flush=True)

# ---------- config ----------
DRY_RUN         = getenv_bool("DRY_RUN", False)
EXCHANGE_ID     = getenv_str("EXCHANGE_ID", "kraken")
API_KEY         = getenv_str("API_KEY", "")
API_SECRET      = getenv_str("API_SECRET", "")
API_PASSWORD    = getenv_str("API_PASSWORD", "")  # some exchanges need it; Kraken ignores
PER_TRADE_USD   = getenv_float("PER_TRADE_USD", 3.0)
DAILY_CAP_USD   = getenv_float("DAILY_CAP_USD", 12.0)
UNIVERSE        = [s.strip().upper() for s in getenv_str("UNIVERSE","BTC/USDT,ETH/USDT").split(",") if s.strip()]
AVOID_REBUY     = getenv_bool("AVOID_REBUY", True)
STOP_LOSS_PCT   = getenv_float("STOP_LOSS_PCT", 0.04)
TAKE_PROFIT_PCT = getenv_float("TAKE_PROFIT_PCT", 0.08)

STATE_DIR       = ".state"
STATE_FILE      = os.path.join(STATE_DIR, "crypto_positions.json")

USD_KEYS        = ("USD","ZUSD")  # Kraken may report ZUSD

# ---------- exchange ----------
def build_exchange() -> ccxt.Exchange:
    klass = getattr(ccxt, EXCHANGE_ID)
    params = {}
    if API_PASSWORD:
        params["password"] = API_PASSWORD
    ex = klass({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        **params,
        "enableRateLimit": True,
        "options": {
            "warnOnFetchOpenOrdersWithoutSymbol": False
        },
    })
    return ex

# ---------- state ----------
def ensure_state() -> Dict[str, Any]:
    os.makedirs(STATE_DIR, exist_ok=True)
    if not os.path.isfile(STATE_FILE):
        data = {"entries":{}, "spent_today":0.0, "day": pd.Timestamp.utcnow().date().isoformat()}
        with open(STATE_FILE, "w") as f: json.dump(data, f)
        return data
    with open(STATE_FILE, "r") as f:
        data = json.load(f)
    # reset spent when day rolls
    today = pd.Timestamp.utcnow().date().isoformat()
    if data.get("day") != today:
        data["day"] = today
        data["spent_today"] = 0.0
    data.setdefault("entries",{})
    return data

def save_state(data: Dict[str,Any]) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

# ---------- helpers ----------
def usd_balance(balances: Dict[str, Any]) -> float:
    # prefer free, then total
    for k in USD_KEYS:
        if k in balances:
            b = balances[k]
            amt = b.get("free", b.get("total", 0.0))
            try: return float(amt)
            except: pass
    return 0.0

def base_from_symbol(symbol: str) -> str:
    # CCXT returns markets; but for simple balance check, base is left side
    return symbol.split("/")[0]

def fetch_price(ex: ccxt.Exchange, symbol: str) -> Optional[float]:
    try:
        t = ex.fetch_ticker(symbol)
        p = t.get("last") or t.get("close")
        return float(p) if p else None
    except Exception as e:
        log(f"[PRICE] {symbol} fail: {e}")
        return None

def market_min_notional(ex: ccxt.Exchange, symbol: str) -> float:
    try:
        m = ex.market(symbol)
        cost = m.get("limits", {}).get("cost", {}).get("min")
        return float(cost) if cost else 0.0
    except Exception:
        return 0.0

def current_base_balance(ex: ccxt.Exchange, base: str) -> float:
    try:
        bal = ex.fetch_balance()
        if base in bal and isinstance(bal[base], dict):
            v = bal[base].get("free", bal[base].get("total", 0.0))
            return float(v)
        # Kraken style sometimes uppercases; handled above already
        return 0.0
    except Exception as e:
        log(f"[BAL] base {base} fail: {e}")
        return 0.0

# ---------- trading actions ----------
def place_market_buy(ex: ccxt.Exchange, symbol: str, notional_usd: float) -> Optional[Tuple[float, float]]:
    price = fetch_price(ex, symbol)
    if not price or price <= 0:
        log(f"[BUY] {symbol} skip (no price)")
        return None

    min_cost = market_min_notional(ex, symbol)
    if min_cost and notional_usd < min_cost:
        log(f"[BUY] {symbol} raise notional {notional_usd} -> min {min_cost}")
        notional_usd = min_cost

    qty = round(notional_usd / price, 6)
    log(f"[BUY] {symbol} qty={qty} price≈{price:.8f} notional≈${notional_usd:.2f}")

    if DRY_RUN:
        return (qty, price)

    try:
        order = ex.create_order(symbol=symbol, type="market", side="buy", amount=qty)
        # best-effort average price
        filled = float(order.get("filled") or qty)
        avg = float(order.get("average") or price)
        log(f"[BUY-OK] {symbol} filled={filled} avg={avg}")
        return (filled, avg)
    except Exception as e:
        log(f"[BUY-FAIL] {symbol}: {e}")
        return None

def place_market_sell_all(ex: ccxt.Exchange, symbol: str, qty: float) -> bool:
    if qty <= 0:
        return False
    log(f"[SELL] {symbol} qty={qty}")
    if DRY_RUN:
        return True
    try:
        ex.create_order(symbol=symbol, type="market", side="sell", amount=qty)
        log(f"[SELL-OK] {symbol}")
        return True
    except Exception as e:
        log(f"[SELL-FAIL] {symbol}: {e}")
        return False

# ---------- main loop ----------
def main() -> None:
    log("=== Crypto Engine Start ===")
    log(f"DRY_RUN={DRY_RUN} EXCHANGE={EXCHANGE_ID} PER_TRADE_USD={PER_TRADE_USD} DAILY_CAP_USD={DAILY_CAP_USD}")
    log(f"UNIVERSE={UNIVERSE} AVOID_REBUY={AVOID_REBUY} SL={STOP_LOSS_PCT} TP={TAKE_PROFIT_PCT}")

    ex = build_exchange()
    state = ensure_state()

    try:
        bal = ex.fetch_balance()
    except Exception as e:
        log(f"[BAL] fetch_balance failed: {e}")
        bal = {}

    usd = usd_balance(bal)
    log(f"USD balance (USD/ZUSD): ${usd:,.2f}")

    # sweep existing positions for TP/SL
    for symbol, entry in list(state["entries"].items()):
        price = fetch_price(ex, symbol)
        if not price:
            continue
        entry_px = float(entry["entry_price"])
        base = base_from_symbol(symbol)
        qty = current_base_balance(ex, base)
        if qty <= 0:
            # position gone; clean it
            log(f"[POS] {symbol} no base balance; removing from state")
            state["entries"].pop(symbol, None)
            continue

        up = (price / entry_px) - 1.0
        if up >= TAKE_PROFIT_PCT:
            log(f"[TP] {symbol} price={price:.8f} entry={entry_px:.8f} gain={up:.2%}")
            if place_market_sell_all(ex, symbol, qty):
                state["entries"].pop(symbol, None)
            continue

        down = 1.0 - (price / entry_px)
        if down >= STOP_LOSS_PCT:
            log(f"[SL] {symbol} price={price:.8f} entry={entry_px:.8f} drawdown={down:.2%}")
            if place_market_sell_all(ex, symbol, qty):
                state["entries"].pop(symbol, None)
            continue

    # save early in case sells modified state
    save_state(state)

    # compute remaining daily budget
    remaining = max(0.0, DAILY_CAP_USD - float(state.get("spent_today", 0.0)))
    log(f"Daily remaining budget: ${remaining:.2f}")
    if remaining < 0.01 or PER_TRADE_USD <= 0:
        log("Nothing to buy (budget exhausted or per-trade notional too small).")
        return

    # place buys across universe (respect AVOID_REBUY and remaining budget)
    for symbol in UNIVERSE:
        if remaining < PER_TRADE_USD:
            log("Budget exhausted for today.")
            break

        if AVOID_REBUY and symbol in state["entries"]:
            log(f"[SKIP] {symbol} (state shows held)")
            continue

        base = base_from_symbol(symbol)
        base_qty = current_base_balance(ex, base)
        if AVOID_REBUY and base_qty > 0:
            log(f"[SKIP] {symbol} (base balance {base_qty} > 0)")
            continue

        res = place_market_buy(ex, symbol, PER_TRADE_USD)
        if not res:
            continue
        filled_qty, avg_px = res
        state["entries"][symbol] = {
            "entry_price": float(avg_px),
            "ts": int(time.time()),
        }
        state["spent_today"] = float(state.get("spent_today", 0.0)) + PER_TRADE_USD
        remaining = max(0.0, DAILY_CAP_USD - state["spent_today"])
        save_state(state)

    log("=== Crypto Engine Done ===")

if __name__ == "__main__":
    main()
