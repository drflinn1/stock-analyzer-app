#!/usr/bin/env python3
# main.py — Crypto live bot with trailing-profit logs and state
#
# - Trail state persists between runs in .state/trails.json
# - Positions persist (entry price / qty) in .state/positions.json (kept by this bot’s own buys)
# - Uses env knobs for everything important
#
# Logs you’ll now see (new):
#   trails_loaded=N symbols on start
#   TRAIL activate SYMBOL at 1.2345 (+3.20% ≥ 3.00%)
#   TRAIL new high SYMBOL anchor 1.3456
#   SELL placed SYMBOL all (trailing-stop drawdown 1.02% ≥ 1.00% from anchor 1.3456)

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

###############################################################################
# Config via environment (provide from GitHub Actions "env:" or repo secrets) #
###############################################################################
EXCHANGE_ID        = os.getenv("EXCHANGE", "kraken")     # "kraken"
DRY_RUN            = os.getenv("DRY_RUN", "true").lower() == "true"

# Universe (comma-separated, CCXT symbols)
UNIVERSE           = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,DOGE/USDT,ADA/USDT,XRP/USDT")
SYMBOLS            = [s.strip() for s in UNIVERSE.split(",") if s.strip()]

# Buy sizing / caps
PER_TRADE_USD      = float(os.getenv("PER_TRADE_USD", "10"))
DAILY_CAP_USD      = float(os.getenv("DAILY_CAP_USD", "15"))

# Buy gate (simple dip gate; keep if you’re using the “buy-on-drop” logic)
BUY_GATE_PCT       = float(os.getenv("BUY_GATE_PCT", "0.0"))   # set >0 to require a dip
MIN_NOTIONAL_USD   = float(os.getenv("MIN_NOTIONAL_USD", "5.0"))

# Trailing-profit controls
TRAIL_ACTIVATE_PCT = float(os.getenv("TRAIL_ACTIVATE_PCT", "3.0"))   # start trailing once P&L ≥ X%
TRAIL_OFFSET_PCT   = float(os.getenv("TRAIL_OFFSET_PCT", "1.0"))     # sell if price falls X% from anchor
# Example: +3.00% activate, 1.00% offset

# API (read from GitHub repo secrets)
API_KEY            = os.getenv("KRAKEN_API_KEY", "")
API_SECRET         = os.getenv("KRAKEN_API_SECRET", "")

# State file locations (committed? No. The workflow will keep them in the workspace;
# add an actions/cache step if you want cross-run persistence.)
STATE_DIR          = Path(".state")
TRAILS_PATH        = STATE_DIR / "trails.json"
POSITIONS_PATH     = STATE_DIR / "positions.json"
SPEND_PATH         = STATE_DIR / "daily_spend.json"  # optional: track spend

###############################################################################
# Helpers: JSON state                                                         #
###############################################################################
def _ensure_state():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not TRAILS_PATH.exists():
        TRAILS_PATH.write_text(json.dumps({}, indent=2))
    if not POSITIONS_PATH.exists():
        POSITIONS_PATH.write_text(json.dumps({}, indent=2))
    if not SPEND_PATH.exists():
        SPEND_PATH.write_text(json.dumps({"date": _today(), "spent_usd": 0.0}, indent=2))

def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text() or "{}")
    except Exception:
        return {}

def _save_json(path: Path, data: Dict[str, Any]):
    path.write_text(json.dumps(data, indent=2))

def _today():
    # GitHub Actions runners are UTC; date-only is fine for daily cap
    return time.strftime("%Y-%m-%d", time.gmtime())

###############################################################################
# Exchange via CCXT                                                           #
###############################################################################
def _make_exchange():
    import ccxt  # imported here so the file loads even without ccxt locally
    klass = getattr(ccxt, EXCHANGE_ID)
    exchange = klass({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        # Kraken: markets load is required before creating orders with symbol precision
    })
    return exchange

def _load_markets_safe(exchange):
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"WARNING: load_markets failed: {e}", flush=True)

def _fetch_ticker_price(exchange, symbol: str) -> float:
    t = exchange.fetch_ticker(symbol)
    # mid price (or last)
    price = t.get("last") or t.get("close") or t.get("bid") or t.get("ask")
    return float(price)

def _market_precision(exchange, symbol: str):
    market = exchange.market(symbol)
    amount_prec = market.get("precision", {}).get("amount", 8)
    price_prec  = market.get("precision", {}).get("price", 8)
    min_cost    = None
    limits = market.get("limits") or {}
    cost = limits.get("cost") or {}
    if isinstance(cost, dict):
        min_cost = cost.get("min")
    return amount_prec, price_prec, min_cost

def _round_amount(amount: float, prec: int) -> float:
    factor = 10 ** prec
    return math.floor(amount * factor) / factor

###############################################################################
# Positions & Trails                                                          #
###############################################################################
# positions.json structure:
# { "BTC/USDT": {"qty": 0.00123, "avg_entry": 62500.0}, ... }
#
# trails.json structure:
# {
#   "BTC/USDT": {
#       "activated": true/false,
#       "anchor": 64500.0,
#       "activate_pct": 3.0,
#       "offset_pct": 1.0
#   }
# }

def load_positions() -> Dict[str, Any]:
    return _load_json(POSITIONS_PATH)

def save_positions(d: Dict[str, Any]):
    _save_json(POSITIONS_PATH, d)

def load_trails() -> Dict[str, Any]:
    return _load_json(TRAILS_PATH)

def save_trails(d: Dict[str, Any]):
    _save_json(TRAILS_PATH, d)

def load_daily_spend() -> Dict[str, Any]:
    return _load_json(SPEND_PATH)

def save_daily_spend(d: Dict[str, Any]):
    _save_json(SPEND_PATH, d)

###############################################################################
# Orders (buy/sell)                                                           #
###############################################################################
def place_market_buy(exchange, symbol: str, usd_amount: float, price_now: float, positions: Dict[str, Any]):
    if usd_amount < MIN_NOTIONAL_USD:
        print(f"SKIP buy {symbol}: amount {usd_amount:.2f} < MIN_NOTIONAL_USD {MIN_NOTIONAL_USD:.2f}")
        return False

    amount_prec, _price_prec, min_cost = _market_precision(exchange, symbol)
    if min_cost and usd_amount < float(min_cost):
        print(f"SKIP buy {symbol}: amount {usd_amount:.2f} < exchange min_cost {float(min_cost):.2f}")
        return False

    base_quote = symbol.split("/")
    if len(base_quote) != 2:
        print(f"ERROR symbol format: {symbol}")
        return False

    qty = usd_amount / price_now
    qty = max(qty, 10**-(amount_prec))  # avoid zero due to rounding
    qty = _round_amount(qty, amount_prec)

    if qty <= 0:
        print(f"ERROR qty<=0 after rounding for {symbol}")
        return False

    if DRY_RUN:
        print(f"BUY (dry)  {symbol} qty {qty} ~ ${usd_amount:.2f} at {price_now:.8f}")
    else:
        try:
            order = exchange.create_market_buy_order(symbol, qty)
            print(f"BUY placed {symbol} qty {qty} ~ ${usd_amount:.2f} at ~{price_now:.8f} (order id {order.get('id')})")
        except Exception as e:
            print(f"ERROR create_market_buy_order {symbol}: {e}")
            return False

    # Update our local positions book (simple running avg over one position)
    pos = positions.get(symbol, {"qty": 0.0, "avg_entry": 0.0})
    old_qty = float(pos.get("qty", 0.0))
    old_avg = float(pos.get("avg_entry", 0.0))

    new_qty = old_qty + qty
    if new_qty <= 0:
        new_avg = 0.0
    elif old_qty <= 0:
        new_avg = price_now
    else:
        new_avg = (old_avg * old_qty + price_now * qty) / new_qty

    positions[symbol] = {"qty": new_qty, "avg_entry": new_avg}
    return True

def place_market_sell_all(exchange, symbol: str, price_now: float, positions: Dict[str, Any]):
    pos = positions.get(symbol)
    if not pos:
        print(f"SKIP sell {symbol}: no local position")
        return False

    qty = float(pos.get("qty", 0.0))
    if qty <= 0:
        print(f"SKIP sell {symbol}: qty<=0")
        return False

    amount_prec, _price_prec, _min_cost = _market_precision(exchange, symbol)
    qty = _round_amount(qty, amount_prec)
    if qty <= 0:
        print(f"SKIP sell {symbol}: qty<=0 after rounding")
        return False

    if DRY_RUN:
        print(f"SELL (dry) {symbol} qty {qty} at ~{price_now:.8f}")
    else:
        try:
            order = exchange.create_market_sell_order(symbol, qty)
            print(f"SELL placed {symbol} qty {qty} at ~{price_now:.8f} (order id {order.get('id')})")
        except Exception as e:
            print(f"ERROR create_market_sell_order {symbol}: {e}")
            return False

    # Clear or reduce local position to 0
    positions[symbol] = {"qty": 0.0, "avg_entry": 0.0}
    return True

###############################################################################
# Trailing-profit engine                                                      #
###############################################################################
def maybe_activate_trail(symbol: str, price_now: float, pos: Dict[str, Any], trail: Dict[str, Any]):
    """
    Activate trailing once unrealized P&L ≥ TRAIL_ACTIVATE_PCT.
    """
    qty = float(pos.get("qty", 0.0))
    entry = float(pos.get("avg_entry", 0.0))
    if qty <= 0 or entry <= 0:
        return False

    pnl_pct = (price_now / entry - 1.0) * 100.0
    if not trail.get("activated", False) and pnl_pct >= TRAIL_ACTIVATE_PCT:
        trail["activated"] = True
        trail["anchor"] = price_now
        trail["activate_pct"] = TRAIL_ACTIVATE_PCT
        trail["offset_pct"] = TRAIL_OFFSET_PCT
        # Log exactly as requested
        print(f"TRAIL activate {symbol} at {price_now:.8f} (+{pnl_pct:.2f}% ≥ {TRAIL_ACTIVATE_PCT:.2f}%)")
        return True
    return False

def update_anchor_if_new_high(symbol: str, price_now: float, trail: Dict[str, Any]):
    """
    If activated and price makes a new high, move anchor up and log.
    """
    if not trail.get("activated", False):
        return False

    anchor = float(trail.get("anchor", 0.0))
    if price_now > anchor:
        trail["anchor"] = price_now
        # Log exactly as requested
        print(f"TRAIL new high {symbol} anchor {price_now:.8f}")
        return True
    return False

def check_trail_stop_and_sell(exchange, symbol: str, price_now: float, trail: Dict[str, Any], positions: Dict[str, Any]):
    """
    If activated and drawdown from anchor ≥ offset, sell all, log the trailing-stop reason.
    """
    if not trail.get("activated", False):
        return False

    anchor = float(trail.get("anchor", 0.0))
    if anchor <= 0:
        return False

    dd_pct = (anchor - price_now) / anchor * 100.0
    if dd_pct >= TRAIL_OFFSET_PCT:
        # Log exactly as requested (include drawdown and from-anchor)
        print(f"SELL placed {symbol} all (trailing-stop drawdown {dd_pct:.2f}% ≥ {TRAIL_OFFSET_PCT:.2f}% from anchor {anchor:.8f})")
        ok = place_market_sell_all(exchange, symbol, price_now, positions)
        # Clear trail no matter what (to avoid duplicate logs); if order failed, next run will see qty>0 and can re-activate
        trail.clear()
        trail["activated"] = False
        return ok
    return False

###############################################################################
# Daily spend                                                                 #
###############################################################################
def daily_remaining() -> float:
    data = load_daily_spend()
    today = _today()
    if data.get("date") != today:
        data = {"date": today, "spent_usd": 0.0}
        save_daily_spend(data)
    return max(0.0, DAILY_CAP_USD - float(data.get("spent_usd", 0.0)))

def add_spend(usd: float):
    data = load_daily_spend()
    today = _today()
    if data.get("date") != today:
        data = {"date": today, "spent_usd": 0.0}
    data["spent_usd"] = float(data.get("spent_usd", 0.0)) + float(usd)
    save_daily_spend(data)

###############################################################################
# (Optional) Simple buy-on-drop gate (kept minimal; you can replace with your
# preferred signal picker).                                                   #
###############################################################################
def allowed_to_buy_now() -> bool:
    # Gate on remaining daily cap
    return daily_remaining() >= max(PER_TRADE_USD, MIN_NOTIONAL_USD)

def maybe_buy(exchange, symbol: str, price_now: float, positions: Dict[str, Any]):
    # Skip if already have a position
    if positions.get(symbol, {}).get("qty", 0.0) > 0:
        return False
    if not allowed_to_buy_now():
        print(f"SKIP buy {symbol}: daily cap reached (remaining ${daily_remaining():.2f})")
        return False

    # Basic gate: if BUY_GATE_PCT==0, always allowed; if >0, require price < recent ref
    # For simplicity here, no historical; you can wire your prior signal logic back in.
    if BUY_GATE_PCT > 0:
        # No reference series in this minimal example, so treat as not met
        print(f"SKIP buy {symbol}: BUY_GATE_PCT={BUY_GATE_PCT:.2f}% gate unmet (no dip ref in minimal logic)")
        return False

    ok = place_market_buy(exchange, symbol, PER_TRADE_USD, price_now, positions)
    if ok:
        add_spend(PER_TRADE_USD)
    return ok

###############################################################################
# Main                                                                        #
###############################################################################
def main():
    print("=== START TRADING OUTPUT ===", flush=True)
    _ensure_state()

    trails = load_trails()
    positions = load_positions()

    print(f"trails_loaded={len(trails)} symbols on start")

    # Exchange
    exchange = _make_exchange()
    _load_markets_safe(exchange)

    # Process each symbol
    for symbol in SYMBOLS:
        try:
            price_now = _fetch_ticker_price(exchange, symbol)
        except Exception as e:
            print(f"ERROR fetch_ticker {symbol}: {e}")
            continue

        pos = positions.get(symbol, {"qty": 0.0, "avg_entry": 0.0})
        trail = trails.get(symbol, {"activated": False})

        # 1) If holding, evaluate trail activation / updates / stops
        if pos.get("qty", 0.0) > 0:
            # Activate if profit threshold met
            maybe_activate_trail(symbol, price_now, pos, trail)

            # If activated, see if a new high occurred (ratchet the anchor)
            update_anchor_if_new_high(symbol, price_now, trail)

            # If activated, check drawdown vs offset and sell-all if hit
            check_trail_stop_and_sell(exchange, symbol, price_now, trail, positions)

        else:
            # Not holding → clear any stale trail state for this symbol
            if trail.get("activated"):
                trail = {"activated": False}

            # Optionally buy (minimal gate here; replace with your existing picker)
            maybe_buy(exchange, symbol, price_now, positions)

        # Persist any trail changes
        trails[symbol] = trail

    # Save state
    save_trails(trails)
    save_positions(positions)

    print("=== END TRADING OUTPUT ===", flush=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"FATAL: {e}")
        sys.exit(1)
