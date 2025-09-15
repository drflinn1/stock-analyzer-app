import os, json, math, time, hashlib
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

DROP_PCT = float(os.getenv("DROP_PCT", "0.8"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "1.2"))
TRAIL_PROFIT_PCT = float(os.getenv("TRAIL_PROFIT_PCT", "0.6"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.0"))   # 0 disables
SELL_FRACTION = float(os.getenv("SELL_FRACTION", "1.0"))

FORCE_SELL_ONCE = os.getenv("FORCE_SELL_ONCE", "").strip()   # "ALL" or comma list: "DOGE,XRP"
FORCE_TOKEN = os.getenv("FORCE_TOKEN", "").strip()            # any string; used to ensure one-time behavior

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
    """Return tradeable (free) balances keyed by currency symbol, e.g., {'XRP':24.0,'DOGE':298.0,'USD':4.58}."""
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

def get_min_cost_usd(market: dict, price: float) -> Optional[float]:
    lim = market.get("limits", {})
    cmin = lim.get("cost", {}).get("min")
    if cmin:
        try:
            return float(cmin)
        except Exception:
            pass
    amin = lim.get("amount", {}).get("min")
    if amin and price:
        try:
            return float(amin) * float(price)
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

# === State helpers ===
def ensure_state_symbol(state: Dict[str, Any], symbol: str, entry: float):
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
    if sold_fraction >= 0.999:
        state["positions"].pop(symbol, None)
    else:
        s["peak"] = s.get("entry", s.get("peak", 0.0))

def reconcile_state_with_balances(state: Dict[str, Any], balances: Dict[str, float]):
    """Drop any symbols from state that have zero balance now (e.g., after manual/forced sell)."""
    have = set()
    for ccy, qty in balances.items():
        if ccy in ("USD",): 
            continue
        have.add(f"{ccy}/USD")
    to_remove = []
    for sym in list(state.get("positions", {}).keys()):
        if sym not in have:
            to_remove.append(sym)
    for sym in to_remove:
        state["positions"].pop(sym, None)
        log(f"{sym}: reconciled — removed from state (no balance).")

# === Buy sizing ===
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

def place_market_sell(exchange, symbol: str, qty: float) -> Tuple[bool, str]:
    if DRY_RUN:
        return True, "[DRY_RUN] sell placed"
    try:
        order = exchange.create_market_sell_order(symbol, qty)
        return True, f"order id {order.get('id','?')}"
    except ccxt.BaseError as e:
        return False, f"order rejected ({str(e)})"

# === Sell rule evaluation ===
def evaluate_sell_rules(symbol: str, price: float, s: Dict[str, Any]) -> Optional[str]:
    entry = s.get("entry", price)
    peak = s.get("peak", entry)
    # Take-profit
    if TAKE_PROFIT_PCT > 0 and price >= entry * (1 + TAKE_PROFIT_PCT/100.0):
        return "TAKE_PROFIT"
    # Trailing-stop (only after above entry)
    if TRAIL_PROFIT_PCT > 0 and peak > entry and price <= peak * (1 - TRAIL_PROFIT_PCT/100.0):
        return "TRAILING_STOP"
    # Optional stop-loss
    if STOP_LOSS_PCT > 0 and price <= entry * (1 - STOP_LOSS_PCT/100.0):
        return "STOP_LOSS"
    return None

# === Force-sell (one time) ===
def parse_force_list(force_value: str, balances: Dict[str, float]) -> List[str]:
    """Return list of symbols 'XXX/USD' to liquidate now."""
    if not force_value:
        return []
    if force_value.upper() == "ALL":
        return [f"{ccy}/USD" for ccy in balances.keys() if ccy not in ("USD",)]
    parts = [p.strip().upper() for p in force_value.split(",") if p.strip()]
    syms = []
    for p in parts:
        if p.endswith("/USD"):
            syms.append(p)
        else:
            syms.append(f"{p}/USD")
    return syms

def force_sell_once(exchange, markets, balances, state):
    if not FORCE_SELL_ONCE:
        return
    token_src = (FORCE_SELL_ONCE + "|" + FORCE_TOKEN).encode("utf-8")
    token = hashlib.sha1(token_src).hexdigest()
    done = state.setdefault("force_done", {})
    if done.get(token):
        log(f"FORCE_SELL: token already executed ({FORCE_TOKEN}); skipping.")
        return

    targets = parse_force_list(FORCE_SELL_ONCE, balances)
    if not targets:
        return
    log(f"FORCE_SELL: one-time liquidation for {targets} (token {FORCE_TOKEN})")

    for sym in targets:
        ccy = sym.split("/")[0]
        qty = float(balances.get(ccy, 0.0))
        if qty <= 0:
            log(f"{sym}: SKIP FORCE SELL — no balance.")
            continue
        if sym not in markets:
            log(f"{sym}: SKIP FORCE SELL — market unavailable.")
            continue
        t = fetch_ticker(exchange, sym)
        if not t or not t.get("bid"):
            log(f"{sym}: SKIP FORCE SELL — no quote.")
            continue
        price = float(t["bid"])
        m = markets[sym]
        sell_qty = round_to_precision(exchange, m, qty * SELL_FRACTION)
        notional = sell_qty * price
        mc = get_min_cost_usd(m, price)
        if mc and notional + 1e-9 < mc:
            log(f"{sym}: SKIP FORCE SELL — below min notional ${notional:.2f} < ${mc:.2f}")
            continue
        ok, info = place_market_sell(exchange, sym, sell_qty)
        if ok:
            log(f"SELL {sym}: FORCE_SELL sold {sell_qty} ~${notional:.2f} ({info})")
            # clear from state
            clear_or_reduce_state(state, sym, 1.0)
        else:
            log(f"{sym}: SKIP FORCE SELL — {info}")
        time.sleep(0.25)

    # mark as executed so it won't repeat
    done[token] = True

# === MAIN ===
def main():
    log("=== START TRADING OUTPUT ===")
    state = load_state()

    ex = connect_exchange()
    markets = ex.load_markets()

    # --- BALANCES + RECONCILE ---
    balances = fetch_balances(ex)
    reconcile_state_with_balances(state, balances)

    # --- ONE-TIME FORCE SELL (if configured and not yet executed) ---
    force_sell_once(ex, markets, balances, state)

    # refresh balances after force sells
    balances = fetch_balances(ex)
    usd_free = float(balances.get("USD", 0.0))

    # --- NORMAL SELL PHASE (self-funding) ---
    for ccy, qty in sorted(balances.items()):
        if ccy in ("USD",):
            continue
        sym = f"{ccy}/USD"
        if sym not in markets:
            continue
        t = fetch_ticker(ex, sym)
        if not t or not t.get("bid"):
            continue
        price = float(t["bid"])
        ensure_state_symbol(state, sym, price)
        update_peak(state, sym, price)
        reason = evaluate_sell_rules(sym, price, state["positions"][sym])
        if not reason:
            continue

        m = markets[sym]
        sell_qty = round_to_precision(ex, m, qty * SELL_FRACTION)
        notional = sell_qty * price
        mc = get_min_cost_usd(m, price)
        if mc and notional + 1e-9 < mc:
            log(f"{sym}: SKIP SELL — below min notional ${notional:.2f} < ${mc:.2f}")
            continue
        if sell_qty <= 0:
            log(f"{sym}: SKIP SELL — qty too small after precision")
            continue

        ok, info = place_market_sell(ex, sym, sell_qty)
        if ok:
            log(f"SELL {sym}: {reason} sold {sell_qty} ~${notional:.2f} ({info})")
            clear_or_reduce_state(state, sym, 1.0 if SELL_FRACTION >= 0.999 else SELL_FRACTION)
        else:
            log(f"{sym}: SKIP SELL — {info}")
        time.sleep(0.25)

    # refresh USD after sells
    usd_free = fetch_free_usd(ex)

    # --- BUY PHASE ---
    universe_syms = pick_universe(ex, markets)
    universe_markets = [markets[s] for s in universe_syms if s in markets]

    daily_left = DAILY_CAP_USD
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
