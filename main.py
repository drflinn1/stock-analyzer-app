#!/usr/bin/env python3
import os, sys, time, uuid, math
from typing import List, Dict, Tuple

# --- Third-party ---
# pip install ccxt pandas numpy
import ccxt

# ==============================
# Env / Config
# ==============================
EXCHANGE_NAME = os.getenv("EXCHANGE", "kraken").lower()

API_KEY   = os.getenv("KRAKEN_API_KEY", "")
API_SECRET= os.getenv("KRAKEN_API_SECRET", "")

# Trading toggles
DRY_RUN = os.getenv("DRY_RUN", "true").strip().lower() == "true"

# One-time force sell: e.g. "DOGE,ADA,XRP" or "DOGE/USD,ADA/USD"
FORCE_SELL_RAW = os.getenv("FORCE_SELL", "").strip()

# Budgets
PER_TRADE_USD = float(os.getenv("PER_TRADE_USD", "15"))
DAILY_CAP_USD = float(os.getenv("DAILY_CAP_USD", "30"))

# Universe size for scanning (top liquid USD pairs)
AUTO_UNIVERSE_SIZE = int(os.getenv("AUTO_UNIVERSE_SIZE", "500"))

# Minimum notional to place an order (Kraken rejects dust; keep >= $0.50)
MIN_NOTIONAL_USD = float(os.getenv("MIN_NOTIONAL_USD", "0.50"))

# Take-profit / trailing stop (kept here for completeness; can be expanded later)
TAKE_PROFIT_PCT  = float(os.getenv("TAKE_PROFIT_PCT", "2.0"))   # not used in this file’s demo loop
TRAIL_PCT        = float(os.getenv("TRAIL_PCT", "1.0"))         # "
STOP_LOSS_PCT    = os.getenv("STOP_LOSS_PCT", "").strip()       # optional

# ==============================
# Helpers
# ==============================
def log(msg: str):
    print(msg, flush=True)

def mk_order_id() -> str:
    return (EXCHANGE_NAME[:2] + "-" + uuid.uuid4().hex[:8]).upper()

def normalize_to_usd_pair(sym: str) -> str:
    s = sym.strip().upper()
    if "/" in s:
        return s
    return f"{s}/USD"

def parse_force_sell_list(raw: str) -> List[str]:
    if not raw:
        return []
    parts = [normalize_to_usd_pair(p) for p in raw.split(",") if p.strip()]
    # Remove dupes, preserve order
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

def load_exchange() -> ccxt.Exchange:
    if EXCHANGE_NAME != "kraken":
        raise RuntimeError("Only Kraken is wired in this build.")
    ex = ccxt.kraken({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
        },
    })
    return ex

def safe_fetch_ticker(ex: ccxt.Exchange, symbol: str) -> Dict:
    try:
        return ex.fetch_ticker(symbol)
    except Exception as e:
        return {}

def last_price(ex: ccxt.Exchange, symbol: str) -> float:
    t = safe_fetch_ticker(ex, symbol)
    if t and "last" in t and t["last"]:
        return float(t["last"])
    # fallback try bid/ask
    if t and "info" in t and isinstance(t["info"], dict):
        info = t["info"]
        for k in ("c", "a", "b"):  # close, ask, bid arrays in Kraken payload
            if k in info and isinstance(info[k], list) and info[k]:
                try:
                    return float(info[k][0])
                except:  # noqa: E722
                    pass
    # absolute fallback
    return 0.0

def place_market_sell(ex: ccxt.Exchange, symbol: str, amount: float) -> Tuple[bool, str]:
    if amount <= 0:
        return False, "amount<=0"
    if DRY_RUN:
        return True, mk_order_id()
    try:
        # Market sell (reduceOnly not needed for spot)
        order = ex.create_order(symbol, "market", "sell", amount)
        oid = order.get("id") or order.get("orderId") or mk_order_id()
        return True, str(oid)
    except Exception as e:
        return False, f"order rejected (kraken {repr(e)})"

def place_market_buy_notional(ex: ccxt.Exchange, symbol: str, notional_usd: float) -> Tuple[bool, str, float]:
    """
    Buy by USD notional -> compute amount = usd / price.
    Returns (ok, order_id, filled_amount)
    """
    px = last_price(ex, symbol)
    if px <= 0:
        return False, "no_price", 0.0
    amount = max(0.0, notional_usd / px)
    if amount <= 0:
        return False, "amount<=0", 0.0
    if DRY_RUN:
        return True, mk_order_id(), amount
    try:
        order = ex.create_order(symbol, "market", "buy", amount)
        oid = order.get("id") or order.get("orderId") or mk_order_id()
        filled = float(order.get("filled") or amount)
        return True, str(oid), filled
    except Exception as e:
        return False, f"order rejected (kraken {repr(e)})", 0.0

# ==============================
# Core Logic
# ==============================
def force_sell_if_requested(ex: ccxt.Exchange):
    targets = parse_force_sell_list(FORCE_SELL_RAW)
    if not targets:
        return

    log("FORCE_SELL: one-time liquidation for " + str(targets) + " (token cleanup)")
    # balances by currency
    try:
        bals = ex.fetch_free_balance()
    except Exception:
        bals = {}

    for symbol in targets:
        base, quote = symbol.split("/")
        free_amt = float(bals.get(base, 0.0))
        if free_amt <= 0:
            log(f"{base}: SKIP FORCE SELL – no balance")
            continue

        px = last_price(ex, symbol)
        notional = free_amt * px if px > 0 else 0.0

        if px <= 0:
            log(f"{symbol}: SKIP FORCE SELL – no price")
            continue
        if notional < MIN_NOTIONAL_USD:
            # Kraken commonly rejects dust < ~$0.50
            log(f"{symbol}: SKIP FORCE SELL – below min notional ${notional:.2f} < ${MIN_NOTIONAL_USD:.2f}")
            continue

        ok, oid_or_err = place_market_sell(ex, symbol, free_amt)
        if ok:
            log(f"SELL {symbol}: FORCE_SELL sold {free_amt:.8f} ~${notional:.2f} (order id {oid_or_err})")
        else:
            log(f"{symbol}: SKIP FORCE SELL – {oid_or_err}")

def discover_auto_universe(ex: ccxt.Exchange) -> List[str]:
    """
    Pick up to AUTO_UNIVERSE_SIZE liquid USD spot pairs.
    Preference: non-stablecoins, has price + volume.
    """
    try:
        markets = ex.load_markets()
    except Exception:
        markets = {}

    symbols = []
    for sym, m in markets.items():
        if not isinstance(m, dict):
            continue
        if "/USD" not in sym:
            continue
        if m.get("spot") is False:
            continue
        base = m.get("base", "")
        if base in {"USDT", "USDC", "DAI", "USD"}:
            continue
        symbols.append(sym)

    # Rough liquidity filter by checking we can fetch a sane price
    liquid = []
    for s in symbols:
        px = last_price(ex, s)
        if px > 0:
            liquid.append(s)
        if len(liquid) >= AUTO_UNIVERSE_SIZE:
            break

    log(f"auto_universe: picked {len(liquid)} of {len(symbols)} candidates → {AUTO_UNIVERSE_SIZE} cap")
    return liquid

def usd_available(ex: ccxt.Exchange) -> float:
    try:
        bals = ex.fetch_free_balance()
        return float(bals.get("USD", 0.0))
    except Exception:
        return 0.0

def run_trader():
    log("=== START TRADING OUTPUT ===")

    # Connect
    ex = load_exchange()

    # 1) One-time force sell (if requested this run)
    force_sell_if_requested(ex)

    # 2) Universe discovery (kept lightweight for Actions)
    universe = discover_auto_universe(ex)
    if not universe:
        log("No universe available; aborting this run.")
        log("=== END TRADING OUTPUT ===")
        return

    # 3) Spending budget for this run
    #    We keep a simple per-run cap. (Daily persistence can be wired later.)
    remaining = DAILY_CAP_USD
    if remaining <= 0:
        log("CAP_REACHED: daily remaining ~$0.00, stopping buys.")
        log("=== END TRADING OUTPUT ===")
        return

    # 4) Simple buy pass: buy first few from universe until cap is used
    #    (You can swap this for your signal logic.)
    for sym in universe:
        if remaining < max(PER_TRADE_USD, MIN_NOTIONAL_USD):
            break

        # Make sure price exists and per-trade notional meets minimum
        px = last_price(ex, sym)
        if px <= 0:
            continue

        notional = max(PER_TRADE_USD, MIN_NOTIONAL_USD)
        if remaining < notional:
            break

        ok, oid_or_err, amt = place_market_buy_notional(ex, sym, notional)
        if ok:
            log(f"BUY {sym}: bought {amt:.6f} ~${notional:.2f} (order id {oid_or_err})")
            remaining -= notional
        else:
            if "rejected" in oid_or_err:
                log(f"{sym}: BUY rejected → {oid_or_err}")
            else:
                log(f"{sym}: BUY skipped → {oid_or_err}")

    if remaining <= 0:
        log("CAP_REACHED: daily remaining ~$0.00, stopping buys.")
    else:
        log(f"Budget left after buys: ${remaining:.2f}")

    log("=== END TRADING OUTPUT ===")


if __name__ == "__main__":
    try:
        run_trader()
    except KeyboardInterrupt:
        log("Interrupted by user.")
        sys.exit(130)
