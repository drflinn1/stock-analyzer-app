#!/usr/bin/env python3
import os, sys, time, uuid, json, math, pathlib
from typing import List, Dict, Tuple

# Third-party
import ccxt

# ==============================
# ENV / CONFIG
# ==============================
EXCHANGE_NAME = os.getenv("EXCHANGE", "kraken").lower()
API_KEY       = os.getenv("KRAKEN_API_KEY", "")
API_SECRET    = os.getenv("KRAKEN_API_SECRET", "")

DRY_RUN = os.getenv("DRY_RUN", "true").strip().lower() == "true"

# Budgets
PER_TRADE_USD   = float(os.getenv("PER_TRADE_USD", "15"))
DAILY_CAP_USD   = float(os.getenv("DAILY_CAP_USD", "60"))  # bumped default a bit
MIN_NOTIONAL_USD= float(os.getenv("MIN_NOTIONAL_USD", "0.50"))

# Universe
AUTO_UNIVERSE_SIZE = int(os.getenv("AUTO_UNIVERSE_SIZE", "500"))

# One-time force sell (comma list of tickers; accepts DOGE or DOGE/USD)
FORCE_SELL_RAW = os.getenv("FORCE_SELL", "").strip()

# Sell rules
TAKE_PROFIT_PCT       = float(os.getenv("TAKE_PROFIT_PCT", "2.0"))   # e.g., 2% above avg price
TRAIL_PCT             = float(os.getenv("TRAIL_PCT", "1.0"))         # sell if drawdown from peak >= this
MIN_PROFIT_TO_TRAIL   = float(os.getenv("MIN_PROFIT_TO_TRAIL", "0.5")) # only arm trail after this profit
STOP_LOSS_PCT_RAW     = os.getenv("STOP_LOSS_PCT", "").strip()       # optional; '' disables
STOP_LOSS_PCT         = float(STOP_LOSS_PCT_RAW) if STOP_LOSS_PCT_RAW else None

STATE_DIR = pathlib.Path(".state")
STATE_PATH = STATE_DIR / "positions.json"

# ==============================
# UTILITIES
# ==============================
def log(msg: str): print(msg, flush=True)

def mk_order_id() -> str:
    return (EXCHANGE_NAME[:2] + "-" + uuid.uuid4().hex[:8]).upper()

def normalize_to_usd_pair(sym: str) -> str:
    s = sym.strip().upper()
    return s if "/" in s else f"{s}/USD"

def parse_force_sell_list(raw: str) -> List[str]:
    if not raw: return []
    seen, out = set(), []
    for p in [normalize_to_usd_pair(x) for x in raw.split(",") if x.strip()]:
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
        "options": {"defaultType": "spot"},
    })
    return ex

def safe_fetch_ticker(ex: ccxt.Exchange, symbol: str) -> Dict:
    try: return ex.fetch_ticker(symbol)
    except Exception: return {}

def last_price(ex: ccxt.Exchange, symbol: str) -> float:
    t = safe_fetch_ticker(ex, symbol)
    if t and isinstance(t.get("last", None), (int,float)):
        return float(t["last"])
    # Kraken payload fallback
    info = t.get("info", {}) if t else {}
    for k in ("c","a","b"):
        arr = info.get(k)
        if isinstance(arr, list) and arr:
            try: return float(arr[0])
            except: pass
    return 0.0

def usd_available(ex: ccxt.Exchange) -> float:
    try:
        bals = ex.fetch_free_balance()
        return float(bals.get("USD", 0.0))
    except Exception:
        return 0.0

def place_market_sell(ex: ccxt.Exchange, symbol: str, amount: float) -> Tuple[bool, str]:
    if amount <= 0: return False, "amount<=0"
    if DRY_RUN:     return True, mk_order_id()
    try:
        order = ex.create_order(symbol, "market", "sell", amount)
        oid = order.get("id") or order.get("orderId") or mk_order_id()
        return True, str(oid)
    except Exception as e:
        return False, f"order rejected (kraken {repr(e)})"

def place_market_buy_notional(ex: ccxt.Exchange, symbol: str, usd: float) -> Tuple[bool, str, float, float]:
    px = last_price(ex, symbol)
    if px <= 0: return False, "no_price", 0.0, 0.0
    qty = max(0.0, usd / px)
    if qty <= 0:  return False, "amount<=0", 0.0, 0.0
    if DRY_RUN:   return True, mk_order_id(), qty, px
    try:
        order = ex.create_order(symbol, "market", "buy", qty)
        oid = order.get("id") or order.get("orderId") or mk_order_id()
        filled = float(order.get("filled") or qty)
        # price from payload if present; otherwise our px
        exec_px = float(order.get("price") or px)
        return True, str(oid), filled, exec_px
    except Exception as e:
        return False, f"order rejected (kraken {repr(e)})", 0.0, 0.0

# ==============================
# STATE (positions)  { "ADA": {"qty":..., "avg":..., "peak":...}, ... }
# ==============================
def load_positions() -> Dict[str, Dict[str, float]]:
    if not STATE_PATH.exists(): return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_positions(pos: Dict[str, Dict[str, float]]):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(pos, f, indent=2, sort_keys=True)

def bump_position_after_buy(positions: Dict[str,Dict[str,float]], base: str, buy_qty: float, exec_px: float):
    p = positions.get(base, {"qty":0.0, "avg":exec_px, "peak":exec_px})
    total_qty = p["qty"] + buy_qty
    if total_qty <= 0: 
        positions.pop(base, None); 
        return
    # volume-weighted average price
    p["avg"] = (p["avg"]*p["qty"] + exec_px*buy_qty) / total_qty if p["qty"]>0 else exec_px
    p["qty"] = total_qty
    # reset/raise peak to new price if higher
    p["peak"] = max(p.get("peak", exec_px), exec_px)
    positions[base] = p

def reduce_position_after_sell(positions: Dict[str,Dict[str,float]], base: str, sell_qty: float, exec_px: float):
    p = positions.get(base)
    if not p: return
    p["qty"] = max(0.0, p["qty"] - sell_qty)
    # After a sell, keep avg as is (conservative). Reset peak to current price.
    p["peak"] = exec_px
    if p["qty"] <= 0.0:
        positions.pop(base, None)
    else:
        positions[base] = p

# ==============================
# CORE FEATURES
# ==============================
def force_sell_if_requested(ex: ccxt.Exchange):
    targets = parse_force_sell_list(FORCE_SELL_RAW)
    if not targets: return

    log("FORCE_SELL: one-time liquidation for " + str(targets) + " (token cleanup)")
    try:
        bals = ex.fetch_free_balance()
    except Exception:
        bals = {}

    for symbol in targets:
        base, quote = symbol.split("/")
        free_amt = float(bals.get(base, 0.0))
        if free_amt <= 0:
            log(f"{base}/USD: SKIP FORCE SELL – no balance")
            continue

        px = last_price(ex, symbol)
        notional = free_amt * px if px > 0 else 0.0
        if px <= 0:
            log(f"{symbol}: SKIP FORCE SELL – no price")
            continue
        if notional < MIN_NOTIONAL_USD:
            log(f"{symbol}: SKIP FORCE SELL – below min notional ${notional:.2f} < ${MIN_NOTIONAL_USD:.2f}")
            continue

        ok, oid = place_market_sell(ex, symbol, free_amt)
        if ok:
            log(f"SELL {symbol}: FORCE_SELL sold {free_amt:.8f} ~${notional:.2f} (order id {oid})")
        else:
            log(f"{symbol}: SKIP FORCE SELL – {oid}")

def discover_auto_universe(ex: ccxt.Exchange) -> List[str]:
    try:
        markets = ex.load_markets()
    except Exception:
        markets = {}
    symbols = []
    for sym, m in markets.items():
        if not isinstance(m, dict): continue
        if "/USD" not in sym:       continue
        if m.get("spot") is False:  continue
        base = m.get("base", "")
        if base in {"USD","USDT","USDC","DAI"}:  # skip stables
            continue
        symbols.append(sym)

    liquid = []
    for s in symbols:
        if last_price(ex, s) > 0:
            liquid.append(s)
        if len(liquid) >= AUTO_UNIVERSE_SIZE:
            break

    log(f"auto_universe: picked {len(liquid)} of {len(symbols)} candidates")
    return liquid

def apply_sell_rules(ex: ccxt.Exchange, positions: Dict[str,Dict[str,float]]) -> float:
    """
    Check each held base against TP/TRAIL/SL; execute sells when triggered.
    Returns total USD freed by sells (approx by last price * qty filled).
    """
    if not positions: return 0.0
    try:
        bals = ex.fetch_free_balance()
    except Exception:
        bals = {}
    freed_usd = 0.0

    for base, p in list(positions.items()):
        qty = float(p.get("qty", 0.0))
        avg = float(p.get("avg", 0.0))
        peak= float(p.get("peak", avg))
        if qty <= 0 or avg <= 0: 
            positions.pop(base, None)
            continue

        symbol = f"{base}/USD"
        px = last_price(ex, symbol)
        if px <= 0: 
            continue

        # Update running peak
        if px > peak:
            p["peak"] = px
            peak = px

        # Decide triggers
        should_sell = False
        reason = ""

        # 1) Take-profit
        if px >= avg * (1.0 + TAKE_PROFIT_PCT/100.0):
            should_sell, reason = True, f"TAKE_PROFIT {TAKE_PROFIT_PCT:.2f}%"

        # 2) Trailing-stop (only if reached minimal profit to arm)
        if not should_sell and peak >= avg * (1.0 + MIN_PROFIT_TO_TRAIL/100.0):
            drawdown = (peak - px) / peak * 100.0
            if drawdown >= TRAIL_PCT:
                should_sell, reason = True, f"TRAIL_STOP {TRAIL_PCT:.2f}% from peak"

        # 3) Optional stop-loss
        if not should_sell and STOP_LOSS_PCT is not None:
            if px <= avg * (1.0 - STOP_LOSS_PCT/100.0):
                should_sell, reason = True, f"STOP_LOSS {STOP_LOSS_PCT:.2f}%"

        if not should_sell:
            # keep position; persist updated peak
            positions[base] = p
            continue

        # Enforce min notional
        notional = qty * px
        if notional < MIN_NOTIONAL_USD:
            log(f"{symbol}: SKIP SELL – below min notional ${notional:.2f} < ${MIN_NOTIONAL_USD:.2f} ({reason})")
            continue

        ok, oid = place_market_sell(ex, symbol, qty)
        if ok:
            log(f"SELL {symbol}: {reason} sold {qty:.8f} ~${notional:.2f} (order id {oid})")
            reduce_position_after_sell(positions, base, qty, px)
            freed_usd += notional
        else:
            log(f"{symbol}: SELL rejected – {oid} ({reason})")

    return freed_usd

# ==============================
# MAIN LOOP
# ==============================
def run_trader():
    print("=== START TRADING OUTPUT ===", flush=True)

    ex = load_exchange()

    # Load persisted positions
    positions = load_positions()

    # One-time forced liquidation if requested
    force_sell_if_requested(ex)

    # Apply rule-based sells on existing positions (TP/TRAIL/SL)
    usd_from_sells = apply_sell_rules(ex, positions)
    if usd_from_sells > 0:
        log(f"REALIZED: freed ~${usd_from_sells:.2f} from sells")

    # Discover trade universe
    universe = discover_auto_universe(ex)
    if not universe:
        log("No universe available; aborting this run.")
        save_positions(positions)
        print("=== END TRADING OUTPUT ===", flush=True)
        return

    # Spending budget (simple per-run; daily persistence can be added later)
    remaining = DAILY_CAP_USD
    if remaining < MIN_NOTIONAL_USD:
        log("CAP_REACHED: daily remaining ~$0.00, stopping buys.")
        save_positions(positions)
        print("=== END TRADING OUTPUT ===", flush=True)
        return

    # Buy pass: take first symbols until cap is used
    for sym in universe:
        if remaining < max(PER_TRADE_USD, MIN_NOTIONAL_USD):
            break

        px = last_price(ex, sym)
        if px <= 0: 
            continue

        notional = max(PER_TRADE_USD, MIN_NOTIONAL_USD)
        if remaining < notional:
            break

        ok, oid_or_err, qty, exec_px = place_market_buy_notional(ex, sym, notional)
        if ok:
            log(f"BUY {sym}: bought {qty:.6f} ~${notional:.2f} (order id {oid_or_err})")
            base = sym.split("/")[0]
            bump_position_after_buy(positions, base, qty, exec_px if exec_px>0 else px)
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

    # Persist state
    save_positions(positions)

    print("=== END TRADING OUTPUT ===", flush=True)

if __name__ == "__main__":
    try:
        run_trader()
    except KeyboardInterrupt:
        log("Interrupted by user.")
        sys.exit(130)
