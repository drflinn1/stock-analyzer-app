#!/usr/bin/env python3
import os, json, time, math, sys, pathlib, logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Third-party
import ccxt

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("crypto-live")

STATE_DIR = pathlib.Path(".state")
BAL_FILE = STATE_DIR / "balances.json"
POS_FILE = STATE_DIR / "positions.json"
BUY_PLAN_FILE = STATE_DIR / "buy_plan.json"

# ---- Config from ENV (with safe defaults) ----
def env_bool(s: str, default: bool) -> bool:
    v = os.getenv(s)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")

DRY_RUN = os.getenv("DRY_RUN", "ON").upper()
MIN_BUY_USD = float(os.getenv("MIN_BUY_USD", "10"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))
MAX_BUYS_PER_RUN = int(os.getenv("MAX_BUYS_PER_RUN", "1"))
UNIVERSE_TOP_K = int(os.getenv("UNIVERSE_TOP_K", "25"))
RESERVE_CASH_PCT = float(os.getenv("RESERVE_CASH_PCT", "5"))
DUST_MIN_USD = float(os.getenv("DUST_MIN_USD", "2"))
ROTATE_WHEN_FULL = env_bool("ROTATE_WHEN_FULL", True)
ROTATE_WHEN_CASH_SHORT = env_bool("ROTATE_WHEN_CASH_SHORT", True)

KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET", "")

# ---- Helpers ----
def ensure_state_dir():
    STATE_DIR.mkdir(parents=True, exist_ok=True)

def write_json(path: pathlib.Path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True))

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def connects_private() -> bool:
    # We consider private usable iff both key/secret present (ccxt will still fail if wrong)
    return bool(KRAKEN_API_KEY and KRAKEN_API_SECRET)

def kraken():
    return ccxt.kraken({
        "apiKey": KRAKEN_API_KEY,
        "secret": KRAKEN_API_SECRET,
        "enableRateLimit": True,
        # 'options': {'fetchBalance': {'method': 'privateGetBalance'},},
    })

def usd_from_balance(bal: Dict) -> float:
    # ccxt standardized balance structure: bal["free"]["USD"], also Kraken uses 'USD'
    free = bal.get("free", {})
    # Some accounts might expose 'ZUSD' in raw; ccxt maps to 'USD' in 'total'/'free' most of the time.
    return safe_float(free.get("USD", 0.0)) + safe_float(free.get("ZUSD", 0.0))

def fetch_balances_and_positions(ex):
    """
    Returns (balances_json, positions_json)
    positions_json = list of {symbol, amount, est_usd}
    """
    bal = ex.fetch_balance()  # private; requires keys
    tickers = ex.fetch_tickers()  # public
    prices = {}
    for sym, t in tickers.items():
        # We want USD (spot) prices only
        if sym.endswith("/USD"):
            prices[sym.split("/")[0]] = safe_float(t.get("last") or t.get("close") or 0.0)

    positions = []
    # Iterate free balances and compute USD value where possible
    for coin, amt in bal.get("free", {}).items():
        amount = safe_float(amt, 0.0)
        if amount <= 0:
            continue
        # Skip stable USD forms as "positions"
        if coin.upper() in ("USD", "ZUSD", "USDT", "USDC"):
            continue
        px = prices.get(coin.upper(), 0.0)
        est_usd = amount * px if px > 0 else 0.0
        positions.append({
            "asset": coin.upper(),
            "amount": amount,
            "est_usd": round(est_usd, 2),
            "px_used": px,
        })

    # Sort by est_usd desc
    positions.sort(key=lambda x: x["est_usd"], reverse=True)
    return bal, positions

def build_buy_plan(ex, usd_cash: float, have_n_positions: int) -> List[Dict]:
    """
    Naive plan: if we have room and enough USD, propose up to MAX_BUYS_PER_RUN tiny entries
    into top USD markets (by ticker availability), allocating MIN_BUY_USD each.
    """
    if usd_cash <= MIN_BUY_USD:
        return []

    # Reserve some cash
    reserve = usd_cash * (RESERVE_CASH_PCT / 100.0)
    budget = max(0.0, usd_cash - reserve)
    if budget < MIN_BUY_USD:
        return []

    room = max(0, MAX_POSITIONS - have_n_positions)
    if room == 0 and not ROTATE_WHEN_FULL:
        return []

    n_buys = min(MAX_BUYS_PER_RUN, room) if room > 0 else (1 if ROTATE_WHEN_FULL else 0)
    if n_buys <= 0:
        return []

    markets = ex.load_markets()
    # Candidate: spot markets quoted in USD
    candidates = [m for m in markets.values()
                  if (m.get("quote") == "USD" and not m.get("contract"))]

    # Take first UNIVERSE_TOP_K alphabetical (placeholder for your smarter ranking)
    candidates = sorted(candidates, key=lambda m: m["symbol"])[:UNIVERSE_TOP_K]
    plan = []
    for m in candidates[:n_buys]:
        sym = m["symbol"]  # e.g., "BTC/USD"
        price = safe_float(ex.fetch_ticker(sym).get("last"), 0.0)
        if price <= 0:
            continue
        qty = round(MIN_BUY_USD / price, m.get("precision", {}).get("amount", 6) or 6)
        if qty <= 0:
            continue
        plan.append({"symbol": sym, "usd": MIN_BUY_USD, "qty": float(qty), "price_ref": price})

    return plan

def place_market_buy(ex, sym: str, qty: float):
    if DRY_RUN == "ON":
        log.info(f"[DRY] BUY {sym} qty={qty}")
        return {"dry": True}
    return ex.create_market_buy_order(sym, qty)

def place_market_sell_all(ex, sym: str, qty: float):
    if DRY_RUN == "ON":
        log.info(f"[DRY] SELL {sym} qty={qty}")
        return {"dry": True}
    return ex.create_market_sell_order(sym, qty)

# ---- Routines ----
def warm_snapshots():
    """
    Always produce .state/balances.json, .state/positions.json, and a minimal buy_plan.json
    so the loop never starts with 'all zeros'.
    """
    ensure_state_dir()
    if connects_private():
        try:
            ex = kraken()
            bal, positions = fetch_balances_and_positions(ex)
            write_json(BAL_FILE, bal)
            write_json(POS_FILE, {"positions": positions})
            usd = usd_from_balance(bal)
            plan = build_buy_plan(ex, usd, len(positions))
            write_json(BUY_PLAN_FILE, {"buy_plan": plan})
            log.info(f"Warm snapshot done. USD={usd:.2f}, positions={len(positions)}, buy_plan={len(plan)}")
            return
        except Exception as e:
            log.warning(f"Warm snapshot (private) failed: {e}")

    # Fallback (no secrets or private failed)
    write_json(BAL_FILE, {"free": {"USD": 0}})
    write_json(POS_FILE, {"positions": []})
    write_json(BUY_PLAN_FILE, {"buy_plan": []})
    log.info("Warm snapshot (public-only fallback) written: USD=0, positions=0, buy_plan=0")

def trade_loop_once():
    """
    Simple one-pass loop:
      - Read snapshots (already warmed)
      - If we have dust positions below DUST_MIN_USD, skip selling (fees risk) — just report
      - If we have cash and room, execute buy_plan (market) using qty computed
      - (Minimal) If any position has est_usd < DUST_MIN_USD, ignore; otherwise no auto-sells here
        (your sell-guard logic can be added back on top)
    """
    ensure_state_dir()

    # Reconnect to exchange
    ex = kraken()

    # Reload live data again right before trading
    bal = {}
    positions = []
    try:
        bal, positions = fetch_balances_and_positions(ex)
    except Exception as e:
        log.error(f"Could not fetch balances/positions: {e}")

    usd = usd_from_balance(bal) if bal else 0.0
    write_json(BAL_FILE, bal or {"free": {"USD": usd}})
    write_json(POS_FILE, {"positions": positions})

    log.info(f"USD cash: {usd:.2f} | positions: {len(positions)}")
    if usd <= MIN_BUY_USD:
        log.info("Not enough USD for a new entry.")
        write_json(BUY_PLAN_FILE, {"buy_plan": []})
        return

    plan = build_buy_plan(ex, usd, len(positions))
    write_json(BUY_PLAN_FILE, {"buy_plan": plan})
    if not plan:
        log.info("No buys planned this run.")
        return

    # Execute plan
    for leg in plan:
        sym = leg["symbol"]
        qty = leg["qty"]
        try:
            res = place_market_buy(ex, sym, qty)
            log.info(f"Placed BUY {sym} qty={qty} result={res}")
        except Exception as e:
            log.error(f"BUY failed for {sym}: {e}")

    # (Optional) Very conservative dust cleanup: none by default (protects tiny balances from fee churn)

# ---- CLI ----------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    warm_only = "--warm-only" in args
    loop_once = "--loop-once" in args

    log.info(f"DRY_RUN={DRY_RUN} | MIN_BUY_USD={MIN_BUY_USD} | MAX_POSITIONS={MAX_POSITIONS} | MAX_BUYS_PER_RUN={MAX_BUYS_PER_RUN}")

    warm_snapshots()
    if warm_only and not loop_once:
        return
    trade_loop_once()

if __name__ == "__main__":
    main()
