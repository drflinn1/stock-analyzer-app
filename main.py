import os, json, math, time
from typing import List, Tuple, Optional, Dict, Any
import ccxt

# === ENV ===
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
EXCHANGE = os.getenv("EXCHANGE", "kraken").lower()

PER_TRADE_USD = float(os.getenv("PER_TRADE_USD", "15"))
DAILY_CAP_USD = float(os.getenv("DAILY_CAP_USD", "30"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "6"))

AUTO_UNIVERSE = os.getenv("AUTO_UNIVERSE", "true").lower() == "true"
UNIVERSE_SIZE = int(os.getenv("UNIVERSE_SIZE", "500"))
MANUAL_SYMBOLS = [s.strip() for s in os.getenv("MANUAL_SYMBOLS", "").split(",") if s.strip()]

DROP_PCT = float(os.getenv("DROP_PCT", "0.8"))  # placeholder gate; buy criteria is minimal here
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "1.2"))
TRAIL_PROFIT_PCT = float(os.getenv("TRAIL_PROFIT_PCT", "0.6"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.0"))   # 0 disables
SELL_FRACTION = float(os.getenv("SELL_FRACTION", "1.0"))   # 0-1 portion to sell on hits

VERBOSE = os.getenv("VERBOSE", "1") == "1"

STATE_PATH = ".state/state.json"   # persisted via Actions cache

def log(msg: str): print(msg, flush=True)

# === Exchange wiring ===
def connect_exchange():
    if EXCHANGE != "kraken":
        raise RuntimeError("Only kraken is wired right now.")
    kwargs = dict(enableRateLimit=True)
    if not DRY_RUN:
        kwargs.update(
            apiKey=os.getenv("KRAKEN_API_KEY", ""),
            secret=os.getenv("KRAKEN_API_SECRET", ""),
        )
    return ccxt.kraken(kwargs)

# === Utilities ===
def load_state() -> Dict[str, Any]:
    try:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log(f"WARN: failed to load state: {e}")
    return {}

def save_state(state: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception as e:
        log(f"WARN: failed to save state: {e}")

def round_to_precision(exchange, market, amount: float) -> float:
    prec = market.get("precision", {}).get("amount", None)
    if prec is not None:
        try:
            return float(exchange.amount_to_precision(market["symbol"], amount))
        except Exception:
            pass
    step = market.get("limits", {}).get("amount", {}).get("min", None)
    if step:
        steps = math.floor(amount / step)
        return max(0.0, steps * step)
    return float(f"{amount:.8f}")

def fetch_free_usd(exchange) -> float:
    try:
        bal = exchange.fetch_balance()
        return float(bal.get("free", {}).get("USD", 0.0))
    except Exception as e:
        log(f"WARN fetch_balance failed: {e}")
        return 0.0

def fetch_balances(exchange) -> Dict[str, float]:
    """Return tradeable (free) balances keyed by currency symbol (e.g., 'XRP', 'DOGE')."""
    out = {}
    try:
        bal = exchange.fetch_balance()
        free = bal.get("free", {}) or {}
        for cur, amt in free.items():
            try:
                if float(amt) > 0:
                    out[cur] = float(amt)
            except Exception:
                continue
    except Exception as e:
        log(f"WARN fetch_balance (detailed) failed: {e}")
    return out

def fetch_ticker(exchange, sym: str) -> Optional[dict]:
    try:
        return exchange.fetch_ticker(sym)
    except Exception as e:
        if VERBOSE:
            log(f"WARN fetch_ticker {sym}: {e}")
        return None

def get_min_cost_usd(market: dict, ask: float) -> Optional[float]:
    lim = market.get("limits", {})
    cmin = lim.get("cost", {}).get("min")
    if cmin:
        try:
            return float(cmin)
        except Exception:
            pass
    amin = lim.get("amount", {}).get("min")
    if amin and ask:
        try:
            return float(amin) * float(ask)
        except Exception:
            pass
    return None

def pick_universe(exchange, markets) -> List[str]:
    symbols = [s for s, m in markets.items() if s.endswith("/USD") and m.get("active", True)]
    symbols.sort()
    if AUTO_UNIVERSE:
        chosen = symbols[: min(UNIVERSE_SIZE, len(symbols))]
        preview = chosen[:5] + (["..."] if len(chosen) > 5 else [])
        log(f"auto_universe: picked {len(chosen)} of {len(symbols)} candidates -> {preview}")
        return chosen
    if MANUAL_SYMBOLS:
        return [s for s in MANUAL_SYMBOLS if s in symbols]
    return ["BTC/USD","ETH/USD","SOL/USD","XRP/USD","DOGE/USD"]

# === BUY ===
def size_buy(exchange, market, ask: float, daily_left: float) -> Tuple[str, float, float]:
    target = min(PER_TRADE_USD, daily_left)
    mc = get_min_cost_usd(market, ask)
    if mc and target + 1e-9 < mc:
        return ("SKIP: below min notional ${:.2f} < ${:.2f}".format(target, mc), 0.0, 0.0)

    qty_raw = target / ask
    qty = round_to_precision(exchange, market, qty_raw)
    notional = qty * ask
    if qty <= 0 or notional < 0.98 * target:
        return ("SKIP: qty too small after precision", 0.0, 0.0)
    return ("OK", qty, notional)

def place_market_buy(exchange, symbol: str, qty: float) -> Tuple[bool, str]:
    if DRY_RUN:
        return True, "[DRY_RUN] buy placed"
    try:
        order = exchange.create_market_buy_order(symbol, qty)
        return True, f"order id {order.get('id','?')}"
    except ccxt.BaseError as e:
        return False, f"order rejected ({str(e)})"

# === SELL ===
def place_market_sell(exchange, symbol: str, qty: float) -> Tuple[bool, str]:
    if DRY_RUN:
        return True, "[DRY_RUN] sell placed"
    try:
        order = exchange.create_market_sell_order(symbol, qty)
        return True, f"order id {order.get('id','?')}"
    except ccxt.BaseError as e:
        return False, f"order rejected ({str(e)})"

def ensure_state_symbol(state: Dict[str, Any], symbol: str, entry: float):
    """Create state for symbol if missing."""
    s = state.setdefault("positions", {}).setdefault(symbol, {})
    if "entry" not in s:
        s["entry"] = entry
    if "peak" not in s:
        s["peak"] = entry

def update_peak(state: Dict[str, Any], symbol: str, price: float):
    s = state.get("positions", {}).get(symbol, None)
    if not s: return
    if price > s.get("peak", s.get("entry", price)):
        s["peak"] = price

def clear_or_reduce_state(state: Dict[str, Any], symbol: str, sold_fraction: float):
    s = state.get("positions", {}).get(symbol, None)
    if not s: return
    if sold_fraction >= 0.999:   # sold everything
        state["positions"].pop(symbol, None)
    else:
        # keep entry, reset peak to current entry to avoid instant resell; simple approach
        s["peak"] = s.get("entry", s.get("peak", 0.0))

def evaluate_sell_rules(symbol: str, price: float, s: Dict[str, Any]) -> Optional[str]:
    """
    Returns reason string if a sell should happen, else None.
    """
    entry = s.get("entry", price)
    peak = s.get("peak", entry)

    # Take-profit
    if TAKE_PROFIT_PCT > 0 and price >= entry * (1 + TAKE_PROFIT_PCT/100.0):
        return "TAKE_PROFIT"

    # Trailing profit (only meaningful when above entry)
    if TRAIL_PROFIT_PCT > 0 and peak > entry and price <= peak * (1 - TRAIL_PROFIT_PCT/100.0):
        return "TRAILING_STOP"

    # Optional stop-loss
    if STOP_LOSS_PCT > 0 and price <= entry * (1 - STOP_LOSS_PCT/100.0):
        return "STOP_LOSS"

    return None

# === MAIN ===
def main():
    log("=== START TRADING OUTPUT ===")
    state = load_state()

    ex = connect_exchange()
    markets = ex.load_markets()

    # --- SELL PHASE (self-funding) ---
    balances = fetch_balances(ex)  # e.g., {"XRP": 24.0, "DOGE": 298.0, "USD": 4.58, ...}
    usd_free = float(balances.get("USD", 0.0))
    # Convert CCY balances to /USD symbols we can trade
    for ccy, qty in sorted(balances.items()):
        if ccy in ("USD",):  # skip cash here
            continue
        symbol = f"{ccy}/USD"
        if symbol not in markets:
            continue  # ignore non-USD markets
        t = fetch_ticker(ex, symbol)
        if not t or not t.get("bid"):
            continue
        price = float(t["bid"])
        ensure_state_symbol(state, symbol, price)
        update_peak(state, symbol, price)
        reason = evaluate_sell_rules(symbol, price, state["positions"][symbol])
        if not reason:
            continue

        # size sell
        m = markets[symbol]
        sell_qty_raw = qty * SELL_FRACTION
        sell_qty = round_to_precision(ex, m, sell_qty_raw)
        notional = sell_qty * price
        mc = get_min_cost_usd(m, price)
        if mc and notional + 1e-9 < mc:
            log(f"{symbol}: SKIP SELL — below min notional ${notional:.2f} < ${mc:.2f}")
            continue
        if sell_qty <= 0:
            log(f"{symbol}: SKIP SELL — qty too small after precision")
            continue

        ok, info = place_market_sell(ex, symbol, sell_qty)
        if ok:
            log(f"SELL {symbol}: {reason} sold {sell_qty} ~${notional:.2f} ({info})")
            sold_fraction = min(1.0, sell_qty_raw / max(qty, 1e-9))
            clear_or_reduce_state(state, symbol, sold_fraction)
        else:
            log(f"{symbol}: SKIP SELL — {info}")

        # small pause for rate limits
        time.sleep(0.25)

    # refresh USD after sells (for buys)
    usd_free = fetch_free_usd(ex)

    # --- BUY PHASE ---
    universe_syms = pick_universe(ex, markets)
    universe_markets = [markets[s] for s in universe_syms if s in markets]

    daily_left = DAILY_CAP_USD
    # Global free-USD guard (fees headroom ~1%)
    if usd_free < PER_TRADE_USD * 1.01:
        log(f"SKIP_ALL_BUYS: free_usd=${usd_free:.2f} < gate=${PER_TRADE_USD*1.01:.2f} — no buy attempts this run.")
        save_state(state)
        log("=== END TRADING OUTPUT ===")
        return

    buys = 0
    for m in universe_markets:
        if daily_left < PER_TRADE_USD * 0.99:
            log(f"CAP_REACHED: daily remaining ~${daily_left:.2f}, stopping buys.")
            break

        sym = m["symbol"]
        t = fetch_ticker(ex, sym)
        if not t or not t.get("ask"):
            log(f"{sym}: SKIP — no quote/ask")
            continue
        ask = float(t["ask"])

        status, qty, notion = size_buy(ex, m, ask, daily_left)
        if status.startswith("SKIP"):
            log(f"{sym}: {status[6:]}")
            continue

        if usd_free < PER_TRADE_USD * 1.01:
            log(f"{sym}: SKIP — low USD ${usd_free:.2f} < gate ${PER_TRADE_USD*1.01:.2f}")
            break

        ok, info = place_market_buy(ex, sym, qty)
        if ok:
            log(f"BUY {sym}: bought {qty} ~${notion:.2f} ({info})")
            buys += 1
            spent = min(PER_TRADE_USD, daily_left)
            daily_left -= spent
            usd_free -= spent
            # Seed state (entry & peak) so sells can trigger later
            ensure_state_symbol(state, sym, ask)
        else:
            log(f"{sym}: SKIP BUY — {info}")

        time.sleep(0.25)

    if buys == 0:
        log("No buys this run (likely min-notional/gates).")

    save_state(state)
    log("=== END TRADING OUTPUT ===")

if __name__ == "__main__":
    main()
