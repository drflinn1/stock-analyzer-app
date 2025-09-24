# trader/crypto_engine.py
from __future__ import annotations
import os, json, time
from typing import Dict, Any, List, Tuple, Optional
import ccxt
import pandas as pd
from decimal import Decimal, ROUND_DOWN

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

DRY_RUN         = getenv_bool("DRY_RUN", False)
EXCHANGE_ID     = getenv_str("EXCHANGE_ID", "kraken")
API_KEY         = getenv_str("API_KEY", "")
API_SECRET      = getenv_str("API_SECRET", "")
API_PASSWORD    = getenv_str("API_PASSWORD", "")
PER_TRADE_USD   = getenv_float("PER_TRADE_USD", 3.0)
DAILY_CAP_USD   = getenv_float("DAILY_CAP_USD", 12.0)
UNIVERSE        = [s.strip().upper() for s in getenv_str("UNIVERSE","BTC/USDT,ETH/USDT").split(",") if s.strip()]
AVOID_REBUY     = getenv_bool("AVOID_REBUY", True)
STOP_LOSS_PCT   = getenv_float("STOP_LOSS_PCT", 0.04)
TAKE_PROFIT_PCT = getenv_float("TAKE_PROFIT_PCT", 0.08)

STATE_DIR       = ".state"
STATE_FILE      = os.path.join(STATE_DIR, "crypto_positions.json")
USD_KEYS        = ("USD","ZUSD")

def build_exchange() -> ccxt.Exchange:
    klass = getattr(ccxt, EXCHANGE_ID)
    params = {"enableRateLimit": True, "options": {"warnOnFetchOpenOrdersWithoutSymbol": False}}
    if API_PASSWORD: params["password"] = API_PASSWORD
    ex = klass({"apiKey": API_KEY, "secret": API_SECRET, **params})
    ex.load_markets()
    return ex

def ensure_state() -> Dict[str, Any]:
    os.makedirs(STATE_DIR, exist_ok=True)
    if not os.path.isfile(STATE_FILE):
        data = {"entries":{}, "spent_today":0.0, "day": pd.Timestamp.utcnow().date().isoformat()}
        with open(STATE_FILE, "w") as f: json.dump(data, f); return data
    with open(STATE_FILE, "r") as f: data = json.load(f)
    today = pd.Timestamp.utcnow().date().isoformat()
    if data.get("day") != today:
        data["day"] = today; data["spent_today"] = 0.0
    data.setdefault("entries",{})
    return data

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

def fetch_price(ex: ccxt.Exchange, symbol: str) -> Optional[float]:
    try:
        t = ex.fetch_ticker(symbol); p = t.get("last") or t.get("close")
        return float(p) if p else None
    except Exception as e:
        log(f"[PRICE] {symbol} fail: {e}"); return None

def market_limits(ex: ccxt.Exchange, symbol: str) -> Tuple[float,float,int,int]:
    """
    Returns (min_cost, min_amount, price_precision, amount_precision)
    """
    m = ex.market(symbol)
    limits = m.get("limits", {})
    min_cost = float(limits.get("cost", {}).get("min") or 0.0)
    min_amount = float(limits.get("amount", {}).get("min") or 0.0)
    pp = int(m.get("precision", {}).get("price", 8))
    ap = int(m.get("precision", {}).get("amount", 8))
    return min_cost, min_amount, pp, ap

def q_round(x: float, decimals: int) -> float:
    q = Decimal(str(x)).quantize(Decimal("1e-%d" % decimals), rounding=ROUND_DOWN)
    return float(q)

def current_base_balance(ex: ccxt.Exchange, base: str) -> float:
    try:
        bal = ex.fetch_balance()
        if base in bal and isinstance(bal[base], dict):
            v = bal[base].get("free", bal[base].get("total", 0.0))
            return float(v)
        return 0.0
    except Exception as e:
        log(f"[BAL] base {base} fail: {e}"); return 0.0

def place_market_buy(ex: ccxt.Exchange, symbol: str, notional_usd: float, remaining: float) -> Optional[Tuple[float, float, float]]:
    price = fetch_price(ex, symbol)
    if not price or price <= 0:
        log(f"[BUY] {symbol} skip (no price)"); return None

    min_cost, min_amount, pp, ap = market_limits(ex, symbol)

    # raise notional to min cost if needed
    notional_needed = max(notional_usd, min_cost or 0.0)

    # compute amount and raise further if min_amount binds
    qty = notional_needed / price
    if min_amount and qty < min_amount:
        qty = min_amount
        notional_needed = qty * price

    # respect remaining daily budget
    if notional_needed > remaining + 1e-9:
        log(f"[BUY] {symbol} need ${notional_needed:.2f} (min), remaining ${remaining:.2f} -> skip")
        return None

    # exchange precision
    qty = max(min_amount or 0.0, q_round(qty, ap))
    if qty <= 0:
        log(f"[BUY] {symbol} qty<=0 after precision -> skip")
        return None

    log(f"[BUY] {symbol} qty={qty} price≈{q_round(price, pp)} notional≈${qty*price:.2f} (min_cost={min_cost or 0}, min_amount={min_amount or 0})")

    if DRY_RUN:
        return (qty, price, notional_needed)

    try:
        order = ex.create_order(symbol=symbol, type="market", side="buy", amount=qty)
        filled = float(order.get("filled") or qty)
        avg = float(order.get("average") or price)
        log(f"[BUY-OK] {symbol} filled={filled} avg={avg}")
        return (filled, avg, notional_needed)
    except Exception as e:
        log(f"[BUY-FAIL] {symbol}: {e}")
        return None

def place_market_sell_all(ex: ccxt.Exchange, symbol: str, qty: float) -> bool:
    if qty <= 0: return False
    log(f"[SELL] {symbol} qty={qty}")
    if DRY_RUN: return True
    try:
        ex.create_order(symbol=symbol, type="market", side="sell", amount=qty)
        log(f"[SELL-OK] {symbol}"); return True
    except Exception as e:
        log(f"[SELL-FAIL] {symbol}: {e}"); return False

def main() -> None:
    log("=== Crypto Engine Start ===")
    log(f"DRY_RUN={DRY_RUN} EXCHANGE={EXCHANGE_ID} PER_TRADE_USD={PER_TRADE_USD} DAILY_CAP_USD={DAILY_CAP_USD}")
    log(f"UNIVERSE={UNIVERSE} AVOID_REBUY={AVOID_REBUY} SL={STOP_LOSS_PCT} TP={TAKE_PROFIT_PCT}")

    ex = build_exchange()
    state = ensure_state()

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
        if up >= TAKE_PROFIT_PCT:
            log(f"[TP] {symbol} p={price:.8f} e={entry_px:.8f} +{up:.2%}")
            if place_market_sell_all(ex, symbol, qty): state["entries"].pop(symbol, None); continue
        down = 1.0 - (price / entry_px)
        if down >= STOP_LOSS_PCT:
            log(f"[SL] {symbol} p={price:.8f} e={entry_px:.8f} -{down:.2%}")
            if place_market_sell_all(ex, symbol, qty): state["entries"].pop(symbol, None); continue

    save_state(state)

    remaining = max(0.0, DAILY_CAP_USD - float(state.get("spent_today", 0.0)))
    log(f"Daily remaining budget: ${remaining:.2f}")
    if remaining < 0.01: log("No budget."); return

    for symbol in UNIVERSE:
        if remaining < 0.01: break
        if AVOID_REBUY and symbol in state["entries"]:
            log(f"[SKIP] {symbol} (state held)"); continue
        base = base_from_symbol(symbol)
        if AVOID_REBUY and current_base_balance(ex, base) > 0:
            log(f"[SKIP] {symbol} (base balance > 0)"); continue

        res = place_market_buy(ex, symbol, PER_TRADE_USD, remaining)
        if not res: continue
        filled_qty, avg_px, spent = res
        state["entries"][symbol] = {"entry_price": float(avg_px), "ts": int(time.time())}
        state["spent_today"] = float(state.get("spent_today", 0.0)) + float(spent)
        remaining = max(0.0, DAILY_CAP_USD - state["spent_today"])
        save_state(state)

    log("=== Crypto Engine Done ===")

if __name__ == "__main__":
    main()
