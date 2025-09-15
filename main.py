#!/usr/bin/env python3
# main.py — Auto-Universe (optimized) + Trailing Profit + Min-notional skip

import json, os, sys, time, math, re
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ── Env ───────────────────────────────────────────────────────────────────────
EXCHANGE_ID        = os.getenv("EXCHANGE", "kraken")
DRY_RUN            = os.getenv("DRY_RUN", "true").lower() == "true"

AUTO_UNIVERSE      = os.getenv("AUTO_UNIVERSE", "false").lower() == "true"
UNIVERSE_TOP_N     = int(os.getenv("UNIVERSE_TOP_N", "5"))
QUOTE              = os.getenv("QUOTE", "USD").upper()
MIN_PRICE_USD      = float(os.getenv("MIN_PRICE_USD", "0.0"))
EXCLUDE_BASES      = [s.strip().upper() for s in os.getenv("EXCLUDE_BASES", "USDT,USDC,DAI,USD,UST,EURT").split(",") if s.strip()]
INCLUDE_BASES      = [s.strip().upper() for s in os.getenv("INCLUDE_BASES", "").split(",") if s.strip()]
UNIVERSE_FALLBACK  = os.getenv("SYMBOLS", "BTC/USD,ETH/USD,DOGE/USD,ADA/USD,XRP/USD")
SYMBOLS_DEFAULT    = [s.strip() for s in UNIVERSE_FALLBACK.split(",") if s.strip()]

PER_TRADE_USD      = float(os.getenv("PER_TRADE_USD", "25"))
DAILY_CAP_USD      = float(os.getenv("DAILY_CAP_USD", "125"))
MIN_NOTIONAL_USD   = float(os.getenv("MIN_NOTIONAL_USD", "5.0"))
BUY_GATE_PCT       = float(os.getenv("BUY_GATE_PCT", "0.0"))
TRAIL_ACTIVATE_PCT = float(os.getenv("TRAIL_ACTIVATE_PCT", "3.0"))
TRAIL_OFFSET_PCT   = float(os.getenv("TRAIL_OFFSET_PCT", "1.0"))

API_KEY            = os.getenv("KRAKEN_API_KEY", "")
API_SECRET         = os.getenv("KRAKEN_API_SECRET", "")

STATE_DIR          = Path(".state")
TRAILS_PATH        = STATE_DIR / "trails.json"
POSITIONS_PATH     = STATE_DIR / "positions.json"
SPEND_PATH         = STATE_DIR / "daily_spend.json"

# ── State helpers ─────────────────────────────────────────────────────────────
def _ensure_state():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not TRAILS_PATH.exists():   TRAILS_PATH.write_text(json.dumps({}, indent=2))
    if not POSITIONS_PATH.exists():POSITIONS_PATH.write_text(json.dumps({}, indent=2))
    if not SPEND_PATH.exists():    SPEND_PATH.write_text(json.dumps({"date": _today(), "spent_usd": 0.0}, indent=2))

def _load_json(p: Path) -> Dict[str, Any]:
    try: return json.loads(p.read_text())
    except Exception: return {}

def _save_json(p: Path, data: Dict[str, Any]): p.write_text(json.dumps(data, indent=2))
def _today(): return time.strftime("%Y-%m-%d", time.gmtime())

# ── CCXT ──────────────────────────────────────────────────────────────────────
def _make_exchange():
    import ccxt
    ex = getattr(ccxt, EXCHANGE_ID)({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 15000,   # 15s per request to avoid long hangs
    })
    return ex

def _load_markets_safe(ex):
    try: ex.load_markets()
    except Exception as e: print(f"WARNING: load_markets failed: {e}", flush=True)

def _fetch_ticker_price(ex, symbol: str) -> float:
    t = ex.fetch_ticker(symbol)
    px = t.get("last") or t.get("close") or t.get("bid") or t.get("ask")
    return float(px)

def _market_precision(ex, symbol: str) -> Tuple[int,int,float,Dict[str,Any]]:
    m = ex.market(symbol)
    amount_prec = (m.get("precision") or {}).get("amount", 8)
    price_prec  = (m.get("precision") or {}).get("price", 8)
    min_cost    = None
    limits = m.get("limits") or {}
    cost = limits.get("cost") or {}
    if isinstance(cost, dict): min_cost = cost.get("min")
    return amount_prec, price_prec, (float(min_cost) if min_cost else None), m

def _round_amount(amount: float, prec: int) -> float:
    factor = 10 ** prec
    return math.floor(amount * factor) / factor

# ── Minimum notional estimate ─────────────────────────────────────────────────
def estimate_min_cost_usd(ex, symbol: str, price_now: float) -> float:
    amt_prec, _pp, min_cost, m = _market_precision(ex, symbol)
    fee_buffer = 0.005  # 0.5%
    headroom   = 0.01
    if min_cost is not None:
        return min_cost * (1 + fee_buffer) + headroom
    amount_min = None
    limits = m.get("limits") or {}
    if isinstance(limits.get("amount"), dict):
        amount_min = limits["amount"].get("min")
    if not amount_min or amount_min <= 0:
        amount_min = 10 ** (-amt_prec)
    est = float(amount_min) * float(price_now)
    return est * (1 + fee_buffer) + headroom

def get_free_usd(ex) -> float:
    try:
        bal = ex.fetch_balance()
        if "free" in bal and isinstance(bal["free"], dict):
            return float(bal["free"].get(QUOTE, 0.0))
        return float((bal.get(QUOTE) or {}).get("free") or 0.0)
    except Exception:
        return float("inf")

# ── Auto-universe (optimized: single ticker call per market) ──────────────────
LEV_TOKEN_PAT = re.compile(r"(UP|DOWN|BEAR|BULL|3L|3S|5L|5S)$", re.IGNORECASE)

def _is_weird_base(base: str) -> bool:
    b = base.upper()
    if b in EXCLUDE_BASES: return True
    if LEV_TOKEN_PAT.search(b): return True
    return False

def pick_universe(ex) -> List[str]:
    _load_markets_safe(ex)
    cands = []
    for s, m in ex.markets.items():
        if m.get("type") not in (None, "spot"):  continue
        if m.get("spot") is False:               continue
        if not m.get("active", True):            continue
        if m.get("quote") != QUOTE:              continue
        base = (m.get("base") or "").upper()
        if INCLUDE_BASES and base not in INCLUDE_BASES: continue
        if _is_weird_base(base):                 continue

        try:
            t = ex.fetch_ticker(s)  # ONE call per symbol
            last = float(t.get("last") or t.get("close") or 0.0)
            qvol = float(t.get("quoteVolume") or 0.0)
            if qvol == 0.0:
                base_vol = float(t.get("baseVolume") or 0.0)
                qvol = base_vol * last
        except Exception:
            last, qvol = 0.0, 0.0

        if MIN_PRICE_USD > 0 and (last <= 0 or last < MIN_PRICE_USD):
            continue

        cands.append((s, qvol))

    cands.sort(key=lambda x: (x[1] or 0.0), reverse=True)
    picked = [s for (s, _v) in cands[:max(1, UNIVERSE_TOP_N)]]
    print(f"auto_universe: picked {len(picked)} of {len(cands)} candidates -> {picked}")
    return picked if picked else SYMBOLS_DEFAULT

# ── Positions / Trails ────────────────────────────────────────────────────────
def load_positions(): return _load_json(POSITIONS_PATH)
def save_positions(d): _save_json(POSITIONS_PATH, d)
def load_trails():     return _load_json(TRAILS_PATH)
def save_trails(d):    _save_json(TRAILS_PATH, d)
def load_daily_spend():return _load_json(SPEND_PATH)
def save_daily_spend(d): _save_json(SPEND_PATH, d)

# ── Orders ────────────────────────────────────────────────────────────────────
def place_market_buy(ex, symbol: str, usd_amount: float, price_now: float, positions: Dict[str, Any]):
    if usd_amount < MIN_NOTIONAL_USD:
        print(f"SKIP buy {symbol}: amount ${usd_amount:.2f} < MIN_NOTIONAL_USD ${MIN_NOTIONAL_USD:.2f}")
        return False
    est_min = estimate_min_cost_usd(ex, symbol, price_now)
    if usd_amount + 1e-9 < max(MIN_NOTIONAL_USD, est_min):
        print(f"SKIP buy {symbol}: per-trade ${usd_amount:.2f} < estimated minimum ${est_min:.2f}")
        return False
    if not DRY_RUN:
        free_usd = get_free_usd(ex)
        if usd_amount > free_usd + 1e-6:
            print(f"SKIP buy {symbol}: insufficient free {QUOTE} (${free_usd:.2f} < ${usd_amount:.2f})")
            return False
    amt_prec, _pp, _mc, _m = _market_precision(ex, symbol)
    qty = usd_amount / price_now if price_now > 0 else 0.0
    qty = max(qty, 10**-(amt_prec))
    qty = _round_amount(qty, amt_prec)
    if qty <= 0: 
        print(f"ERROR qty<=0 after rounding for {symbol}")
        return False

    if DRY_RUN:
        print(f"BUY (dry)  {symbol} qty {qty} ~ ${usd_amount:.2f} at {price_now:.8f}")
    else:
        try:
            order = ex.create_market_buy_order(symbol, qty)
            print(f"BUY placed {symbol} qty {qty} ~ ${usd_amount:.2f} at ~{price_now:.8f} (order id {order.get('id')})")
        except Exception as e:
            print(f"ERROR create_market_buy_order {symbol}: {e}")
            return False

    pos = positions.get(symbol, {"qty": 0.0, "avg_entry": 0.0})
    old_q, old_px = float(pos["qty"]), float(pos["avg_entry"])
    new_q = old_q + qty
    new_px = price_now if old_q <= 0 else (old_px*old_q + price_now*qty) / new_q
    positions[symbol] = {"qty": new_q, "avg_entry": new_px}
    return True

def place_market_sell_all(ex, symbol: str, price_now: float, positions: Dict[str, Any]):
    pos = positions.get(symbol)
    if not pos: 
        print(f"SKIP sell {symbol}: no local position"); return False
    amt_prec, _pp, _mc, _m = _market_precision(ex, symbol)
    qty = _round_amount(float(pos.get("qty", 0.0)), amt_prec)
    if qty <= 0: 
        print(f"SKIP sell {symbol}: qty<=0"); return False

    if DRY_RUN:
        print(f"SELL (dry) {symbol} qty {qty} at ~{price_now:.8f}")
    else:
        try:
            order = ex.create_market_sell_order(symbol, qty)
            print(f"SELL placed {symbol} qty {qty} at ~{price_now:.8f} (order id {order.get('id')})")
        except Exception as e:
            print(f"ERROR create_market_sell_order {symbol}: {e}")
            return False

    positions[symbol] = {"qty": 0.0, "avg_entry": 0.0}
    return True

# ── Trailing profit ───────────────────────────────────────────────────────────
def maybe_activate_trail(symbol: str, price_now: float, pos: Dict[str, Any], trail: Dict[str, Any]):
    qty = float(pos.get("qty", 0.0)); entry = float(pos.get("avg_entry", 0.0))
    if qty <= 0 or entry <= 0: return False
    pnl_pct = (price_now/entry - 1.0) * 100.0
    if not trail.get("activated", False) and pnl_pct >= TRAIL_ACTIVATE_PCT:
        trail.update({"activated": True, "anchor": price_now,
                      "activate_pct": TRAIL_ACTIVATE_PCT, "offset_pct": TRAIL_OFFSET_PCT})
        print(f"TRAIL activate {symbol} at {price_now:.8f} (+{pnl_pct:.2f}% ≥ {TRAIL_ACTIVATE_PCT:.2f}%)")
        return True
    return False

def update_anchor_if_new_high(symbol: str, price_now: float, trail: Dict[str, Any]):
    if not trail.get("activated", False): return False
    if price_now > float(trail.get("anchor", 0.0)):
        trail["anchor"] = price_now
        print(f"TRAIL new high {symbol} anchor {price_now:.8f}")
        return True
    return False

def check_trail_stop_and_sell(ex, symbol: str, price_now: float, trail: Dict[str, Any], positions: Dict[str, Any]):
    if not trail.get("activated", False): return False
    anchor = float(trail.get("anchor", 0.0))
    if anchor <= 0: return False
    dd_pct = (anchor - price_now) / anchor * 100.0
    if dd_pct >= TRAIL_OFFSET_PCT:
        print(f"SELL placed {symbol} all (trailing-stop drawdown {dd_pct:.2f}% ≥ {TRAIL_OFFSET_PCT:.2f}% from anchor {anchor:.8f})")
        ok = place_market_sell_all(ex, symbol, price_now, positions)
        trail.clear(); trail["activated"] = False
        return ok
    return False

# ── Daily spend & gate ────────────────────────────────────────────────────────
def daily_remaining():
    d = load_daily_spend(); today = _today()
    if d.get("date") != today:
        d = {"date": today, "spent_usd": 0.0}; save_daily_spend(d)
    return max(0.0, DAILY_CAP_USD - float(d.get("spent_usd", 0.0)))

def add_spend(usd: float):
    d = load_daily_spend(); today = _today()
    if d.get("date") != today: d = {"date": today, "spent_usd": 0.0}
    d["spent_usd"] = float(d.get("spent_usd", 0.0)) + float(usd); save_daily_spend(d)

def allowed_to_buy_now(): return daily_remaining() >= max(PER_TRADE_USD, MIN_NOTIONAL_USD)

def maybe_buy(ex, symbol: str, price_now: float, positions: Dict[str, Any]):
    if positions.get(symbol, {}).get("qty", 0.0) > 0: return False
    if not allowed_to_buy_now():
        print(f"SKIP buy {symbol}: daily cap reached (remaining ${daily_remaining():.2f})"); return False
    if BUY_GATE_PCT > 0:
        print(f"SKIP buy {symbol}: BUY_GATE_PCT={BUY_GATE_PCT:.2f}% gate unmet (no dip ref in minimal logic)"); return False
    if place_market_buy(ex, symbol, PER_TRADE_USD, price_now, positions):
        add_spend(PER_TRADE_USD); return True
    return False

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== START TRADING OUTPUT ===", flush=True)
    _ensure_state()
    trails = load_trails(); positions = load_positions()

    ex = _make_exchange()
    symbols = pick_universe(ex) if AUTO_UNIVERSE else SYMBOLS_DEFAULT

    print(f"trails_loaded={len(trails)} symbols on start")
    for symbol in symbols:
        try:
            price = _fetch_ticker_price(ex, symbol)
        except Exception as e:
            print(f"ERROR fetch_ticker {symbol}: {e}"); continue

        pos = positions.get(symbol, {"qty": 0.0, "avg_entry": 0.0})
        trail = trails.get(symbol, {"activated": False})

        if pos.get("qty", 0.0) > 0:
            maybe_activate_trail(symbol, price, pos, trail)
            update_anchor_if_new_high(symbol, price, trail)
            check_trail_stop_and_sell(ex, symbol, price, trail, positions)
        else:
            if trail.get("activated"): trail = {"activated": False}
            maybe_buy(ex, symbol, price, positions)

        trails[symbol] = trail

    save_trails(trails); save_positions(positions)
    print("=== END TRADING OUTPUT ===", flush=True)

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("Interrupted"); sys.exit(130)
    except Exception as e: print(f"FATAL: {e}"); sys.exit(1)
