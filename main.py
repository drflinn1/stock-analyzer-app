#!/usr/bin/env python3
import os, json, time, math, sys, pathlib, logging
from typing import Dict, List, Tuple

import ccxt

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("crypto-live")

# ---------------- Paths ----------------
STATE_DIR = pathlib.Path(".state")
BAL_FILE = STATE_DIR / "balances.json"
POS_FILE = STATE_DIR / "positions.json"
BUY_PLAN_FILE = STATE_DIR / "buy_plan.json"
ENTRY_FILE = STATE_DIR / "entries.json"      # tracks average entry and qty
HIGHWATER_FILE = STATE_DIR / "high_water.json"  # per-asset session high for trailing

def ensure_state_dir():
    STATE_DIR.mkdir(parents=True, exist_ok=True)

def read_json(path: pathlib.Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def write_json(path: pathlib.Path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True))

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")

# ---------------- Config (env) ----------------
DRY_RUN = os.getenv("DRY_RUN", "ON").upper()

MIN_BUY_USD = float(os.getenv("MIN_BUY_USD", "10"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))
MAX_BUYS_PER_RUN = int(os.getenv("MAX_BUYS_PER_RUN", "1"))
UNIVERSE_TOP_K = int(os.getenv("UNIVERSE_TOP_K", "25"))
RESERVE_CASH_PCT = float(os.getenv("RESERVE_CASH_PCT", "5"))
DUST_MIN_USD = float(os.getenv("DUST_MIN_USD", "2"))
ROTATE_WHEN_FULL = env_bool("ROTATE_WHEN_FULL", True)
ROTATE_WHEN_CASH_SHORT = env_bool("ROTATE_WHEN_CASH_SHORT", True)

# --- Sell Guard knobs (the guard workflow looks for these exact tokens) ---
# TAKE_PROFIT / take_profit
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "12"))
# TRAIL / trailing
TRAIL_PCT = float(os.getenv("TRAIL_PCT", "8"))
# STOP_LOSS / stop_loss
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "10"))

MIN_SELL_USD = float(os.getenv("MIN_SELL_USD", "10"))

KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET", "")

def connects_private() -> bool:
    return bool(KRAKEN_API_KEY and KRAKEN_API_SECRET)

def kraken():
    return ccxt.kraken({
        "apiKey": KRAKEN_API_KEY,
        "secret": KRAKEN_API_SECRET,
        "enableRateLimit": True,
    })

def usd_from_balance(bal: Dict) -> float:
    free = bal.get("free", {})
    return safe_float(free.get("USD", 0.0)) + safe_float(free.get("ZUSD", 0.0))

def fetch_balances_and_positions(ex):
    bal = ex.fetch_balance()  # private
    tickers = ex.fetch_tickers()  # public
    prices = {}
    for sym, t in tickers.items():
        if sym.endswith("/USD"):
            prices[sym.split("/")[0]] = safe_float(t.get("last") or t.get("close") or 0.0)

    positions = []
    for coin, amt in bal.get("free", {}).items():
        amount = safe_float(amt, 0.0)
        if amount <= 0:
            continue
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
    positions.sort(key=lambda x: x["est_usd"], reverse=True)
    return bal, positions

def build_buy_plan(ex, usd_cash: float, have_n_positions: int) -> List[Dict]:
    if usd_cash <= MIN_BUY_USD:
        return []
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
    candidates = [m for m in markets.values() if (m.get("quote") == "USD" and not m.get("contract"))]
    candidates = sorted(candidates, key=lambda m: m["symbol"])[:UNIVERSE_TOP_K]

    plan = []
    for m in candidates[:n_buys]:
        sym = m["symbol"]  # "BTC/USD"
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

# ---------------- Sell Guard (simple + transparent) ----------------
def symbol_for_asset(ex, asset: str) -> str:
    """Map 'BTC' -> 'BTC/USD' if available."""
    sym = f"{asset}/USD"
    markets = ex.markets or ex.load_markets()
    return sym if sym in markets else ""

def update_entry(entries: Dict, asset: str, buy_px: float, buy_qty: float):
    e = entries.get(asset)
    if not e:
        entries[asset] = {"avg_entry": buy_px, "qty": buy_qty}
        return
    # weighted average
    prev_px, prev_qty = e["avg_entry"], e["qty"]
    new_qty = prev_qty + buy_qty
    if new_qty <= 0:
        entries[asset] = {"avg_entry": buy_px, "qty": buy_qty}
    else:
        new_avg = (prev_px * prev_qty + buy_px * buy_qty) / new_qty
        entries[asset] = {"avg_entry": new_avg, "qty": new_qty}

def sell_guard_decision(asset: str, last_px: float, entry_px: float, high_px: float) -> str:
    """
    Returns one of: "", "TAKE_PROFIT", "TRAIL", "STOP_LOSS"
    - TAKE_PROFIT when gain >= TAKE_PROFIT_PCT
    - TRAIL when drop from high >= TRAIL_PCT
    - STOP_LOSS when loss >= STOP_LOSS_PCT
    """
    if last_px <= 0 or entry_px <= 0:
        return ""

    pnl_pct = (last_px - entry_px) / entry_px * 100.0
    if pnl_pct >= TAKE_PROFIT_PCT:
        return "TAKE_PROFIT"  # token required by the guard workflow

    # keep high-water up to date externally; decision just checks the gap
    if high_px > 0:
        drop_from_high = (high_px - last_px) / high_px * 100.0
        if drop_from_high >= TRAIL_PCT:
            return "TRAIL"  # token required

    if pnl_pct <= -STOP_LOSS_PCT:
        return "STOP_LOSS"  # token required

    return ""

# ---------------- Warm snapshots ----------------
def warm_snapshots():
    ensure_state_dir()
    if connects_private():
        try:
            ex = kraken()
            bal, positions = fetch_balances_and_positions(ex)
            write_json(BAL_FILE, bal)
            write_json(POS_FILE, {"positions": positions})
            usd = usd_from_balance(bal)
            # Minimal plan preview
            plan = build_buy_plan(ex, usd, len(positions))
            write_json(BUY_PLAN_FILE, {"buy_plan": plan})
            # initialize entries/high-water files if missing
            if not ENTRY_FILE.exists():
                write_json(ENTRY_FILE, {})
            if not HIGHWATER_FILE.exists():
                write_json(HIGHWATER_FILE, {})
            log.info(f"Warm snapshot done. USD={usd:.2f}, positions={len(positions)}, buy_plan={len(plan)}")
            return
        except Exception as e:
            log.warning(f"Warm snapshot (private) failed: {e}")

    write_json(BAL_FILE, {"free": {"USD": 0}})
    write_json(POS_FILE, {"positions": []})
    write_json(BUY_PLAN_FILE, {"buy_plan": []})
    if not ENTRY_FILE.exists(): write_json(ENTRY_FILE, {})
    if not HIGHWATER_FILE.exists(): write_json(HIGHWATER_FILE, {})
    log.info("Warm snapshot (public-only fallback) written: USD=0, positions=0, buy_plan=0")

# ---------------- Loop ----------------
def trade_loop_once():
    ensure_state_dir()
    ex = kraken()

    # Live refresh
    try:
        bal, positions = fetch_balances_and_positions(ex)
    except Exception as e:
        log.error(f"Could not fetch balances/positions: {e}")
        bal, positions = {}, []
    usd = usd_from_balance(bal) if bal else 0.0
    write_json(BAL_FILE, bal or {"free": {"USD": usd}})
    write_json(POS_FILE, {"positions": positions})

    entries = read_json(ENTRY_FILE, {})
    highw = read_json(HIGHWATER_FILE, {})

    log.info(f"USD cash: {usd:.2f} | positions: {len(positions)}")

    # ---------- SELL PHASE (guard) ----------
    for p in positions:
        asset = p["asset"]
        amt = safe_float(p["amount"], 0.0)
        est_usd = safe_float(p["est_usd"], 0.0)
        if amt <= 0 or est_usd < MIN_SELL_USD:
            continue
        sym = symbol_for_asset(ex, asset)
        if not sym:
            continue
        last_px = safe_float(ex.fetch_ticker(sym).get("last"), 0.0)

        e = entries.get(asset, {})
        entry_px = safe_float(e.get("avg_entry", 0.0), 0.0)
        # maintain high-water
        prev_high = safe_float(highw.get(asset, 0.0), 0.0)
        highw[asset] = max(prev_high, last_px)

        decision = sell_guard_decision(asset, last_px, entry_px, highw.get(asset, last_px))
        if decision:
            log.info(f"[SELL-GUARD] {asset} {decision} | last={last_px:.6f} entry={entry_px:.6f} high={highw.get(asset,0.0):.6f}")
            try:
                res = place_market_sell_all(ex, sym, amt)
                log.info(f"Placed SELL {sym} qty={amt} result={res}")
                # clear tracking on full exit
                entries.pop(asset, None)
                highw.pop(asset, None)
            except Exception as se:
                log.error(f"SELL failed for {sym}: {se}")

    # persist any updated highs
    write_json(HIGHWATER_FILE, highw)
    write_json(ENTRY_FILE, entries)

    # ---------- BUY PHASE ----------
    if usd > MIN_BUY_USD:
        plan = build_buy_plan(ex, usd, len(positions))
    else:
        plan = []
    write_json(BUY_PLAN_FILE, {"buy_plan": plan})

    for leg in plan:
        sym = leg["symbol"]
        qty = leg["qty"]
        asset = sym.split("/")[0]
        px = safe_float(ex.fetch_ticker(sym).get("last"), 0.0)
        try:
            res = place_market_buy(ex, sym, qty)
            log.info(f"Placed BUY {sym} qty={qty} result={res}")
            # update entry average & highwater
            update_entry(entries, asset, px, qty)
            # initialize high-water at buy price if higher
            highw[asset] = max(highw.get(asset, 0.0), px)
        except Exception as be:
            log.error(f"BUY failed for {sym}: {be}")

    write_json(ENTRY_FILE, entries)
    write_json(HIGHWATER_FILE, highw)

# ---------------- CLI ----------------
def main():
    args = sys.argv[1:]
    warm_only = "--warm-only" in args
    loop_once = "--loop-once" in args
    log.info(
        "DRY_RUN=%s | MIN_BUY_USD=%.2f | MAX_POSITIONS=%d | MAX_BUYS_PER_RUN=%d | "
        "TAKE_PROFIT_PCT=%.2f | TRAIL_PCT=%.2f | STOP_LOSS_PCT=%.2f",
        DRY_RUN, MIN_BUY_USD, MAX_POSITIONS, MAX_BUYS_PER_RUN,
        TAKE_PROFIT_PCT, TRAIL_PCT, STOP_LOSS_PCT
    )
    warm_snapshots()
    if warm_only and not loop_once:
        return
    trade_loop_once()

if __name__ == "__main__":
    main()
