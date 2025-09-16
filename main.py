#!/usr/bin/env python3
import os, time, json, math, re, sys, random, datetime
from collections import defaultdict

# =========================
# Config via environment
# =========================
EXCHANGE_NAME   = os.getenv("EXCHANGE", "kraken").lower()
MODE            = os.getenv("MODE", "live").lower()              # "live" or "dry"
DRY_RUN         = MODE != "live" or os.getenv("DRY_RUN", "false").lower() == "true"

PER_TRADE_USD   = float(os.getenv("PER_TRADE_USD", "10"))
DAILY_CAP_USD   = float(os.getenv("DAILY_CAP_USD", "25"))        # workflow may persist/override this
DROP_PCT        = float(os.getenv("DROP_PCT", "2.0"))

TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "2.0"))     # e.g., 2.0%
TRAIL_ACTIVATE  = float(os.getenv("TRAILING_ACTIVATE_PCT", "1.0"))
TRAIL_DELTA     = float(os.getenv("TRAILING_DELTA_PCT", "1.0"))
STOP_LOSS_PCT   = float(os.getenv("STOP_LOSS_PCT", "0"))         # 0 disables SL

MAX_POSITIONS   = int(os.getenv("MAX_POSITIONS", "50"))          # cap total concurrent holdings
PER_ASSET_CAP   = float(os.getenv("PER_ASSET_CAP_USD", "50"))    # cap USD exposure per symbol
COOLDOWN_MIN    = int(os.getenv("COOLDOWN_MINUTES", "30"))       # after a SELL, wait N minutes before rebuy

AUTO_UNIV_COUNT = int(os.getenv("AUTO_UNIVERSE_COUNT", "500"))   # number of candidates from auto picker
QUOTE           = os.getenv("QUOTE", "USD").upper()

STATE_DIR       = ".state"
POSITIONS_FILE  = os.path.join(STATE_DIR, "positions.json")      # our simple ledger of entries
COOLDOWN_FILE   = os.path.join(STATE_DIR, "cooldown.json")       # last-sell timestamps per symbol

os.makedirs(STATE_DIR, exist_ok=True)

def _load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

positions = _load_json(POSITIONS_FILE, {})     # sym -> list of lots [{qty, cost, entry}]
cooldown  = _load_json(COOLDOWN_FILE, {})      # sym -> iso timestamp of last SELL

# =========================
# Minimal market adapter (CCXT)
# =========================
try:
    import ccxt  # type: ignore
except Exception as e:
    print("ccxt not available; install in requirements.txt", file=sys.stderr)
    ccxt = None

def mk_exchange():
    kwargs = {}
    if EXCHANGE_NAME == "kraken":
        kwargs = {
            "apiKey": os.getenv("KRAKEN_API_KEY", ""),
            "secret": os.getenv("KRAKEN_API_SECRET", ""),
            "enableRateLimit": True,
        }
        ex = ccxt.kraken(kwargs)
    else:
        raise RuntimeError(f"Unsupported exchange: {EXCHANGE_NAME}")
    ex.load_markets()
    return ex

exchange = mk_exchange() if ccxt else None

def get_free_usd():
    if not exchange:
        return 0.0
    bal = exchange.fetch_free_balance()
    return float(bal.get(QUOTE, 0.0))

def market_price(symbol):  # e.g., "ACH/USD"
    if not exchange: return 0.0
    ticker = exchange.fetch_ticker(symbol)
    return float(ticker["last"] or ticker["close"] or 0.0)

def list_tradeable_pairs():
    # pick spot markets with desired quote
    markets = []
    for m in exchange.markets.values():
        if m.get("spot") and m.get("active") and m.get("quote") == QUOTE:
            markets.append(m["symbol"])
    return sorted(set(markets))

# =========================
# Auto-universe (simple volume proxy)
# =========================
def autopick_universe(limit=AUTO_UNIV_COUNT):
    symbols = list_tradeable_pairs()
    # crude proxy ranking by base volume if provided; else random shuffle
    def score(sym):
        try:
            t = exchange.fetch_ticker(sym)
            return float(t.get("baseVolume") or 0.0)
        except Exception:
            return 0.0
    ranked = sorted(symbols, key=score, reverse=True)
    if len(ranked) < limit:
        ranked += [s for s in symbols if s not in ranked]
    pick = ranked[:limit]
    print(f"auto_universe: picked {len(pick)} of {len(symbols)} candidates")
    return pick

# =========================
# Ledger helpers
# =========================
def total_positions_count():
    return sum(1 for lots in positions.values() for _ in lots)

def position_usd_exposure(sym, price):
    lots = positions.get(sym, [])
    qty  = sum(l["qty"] for l in lots)
    return qty * price

def add_lot(sym, qty, cost):
    lots = positions.setdefault(sym, [])
    lots.append({"qty": qty, "cost": cost, "entry": datetime.datetime.utcnow().isoformat()})
    _save_json(POSITIONS_FILE, positions)

def reduce_or_clear(sym):
    positions.pop(sym, None)
    _save_json(POSITIONS_FILE, positions)

def cooldown_active(sym):
    if sym not in cooldown: return False
    t = datetime.datetime.fromisoformat(cooldown[sym])
    return (datetime.datetime.utcnow() - t) < datetime.timedelta(minutes=COOLDOWN_MIN)

def mark_sold(sym):
    cooldown[sym] = datetime.datetime.utcnow().isoformat()
    _save_json(COOLDOWN_FILE, cooldown)

# =========================
# Order emulation / live
# =========================
def place_buy(sym, usd_amt):
    px = market_price(sym)
    if px <= 0: return False, 0.0, "bad_price"
    qty = usd_amt / px
    if DRY_RUN:
        print(f"BUY {sym}: DRY ~${usd_amt:.2f} @ ~{px:.8f} qty~{qty:.8f}")
        add_lot(sym, qty, px)
        return True, qty, "dry"
    # live
    try:
        order = exchange.create_market_buy_order(sym, qty)
        print(f"BUY {sym}: bought {qty:.6f} ~${usd_amt:.2f} (order id {order.get('id','?')})")
        add_lot(sym, qty, px)
        return True, qty, "ok"
    except Exception as e:
        print(f"BUY ERROR {sym}: {e}")
        return False, 0.0, "err"

def place_sell(sym, qty, reason):
    px = market_price(sym)
    if px <= 0: return False, 0.0, "bad_price"
    usd = qty * px
    if DRY_RUN:
        print(f"SELL {sym}: {reason} sold {qty:.6f} ~${usd:.2f}")
        reduce_or_clear(sym)
        mark_sold(sym)
        return True, usd, "dry"
    try:
        order = exchange.create_market_sell_order(sym, qty)
        print(f"SELL {sym}: {reason} sold {qty:.6f} ~${usd:.2f} (order id {order.get('id','?')})")
        reduce_or_clear(sym)
        mark_sold(sym)
        return True, usd, "ok"
    except Exception as e:
        print(f"SELL ERROR {sym}: {e}")
        return False, 0.0, "err"

# =========================
# Sell engine (TP / Trailing / SL)
# =========================
def evaluate_sells():
    realized = 0.0
    for sym, lots in list(positions.items()):
        if not lots: continue
        qty = sum(l["qty"] for l in lots)
        avg_cost = sum(l["qty"]*l["cost"] for l in lots)/qty
        px = market_price(sym)
        if px <= 0: continue
        chg_pct = (px - avg_cost) / avg_cost * 100.0

        # Take-profit
        if TAKE_PROFIT_PCT > 0 and chg_pct >= TAKE_PROFIT_PCT:
            ok, usd, _ = place_sell(sym, qty, f"TAKE_PROFIT {TAKE_PROFIT_PCT:.2f}%")
            realized += usd
            continue

        # Trailing: when activated above threshold, sell if pullback >= TRAIL_DELTA
        if TRAIL_ACTIVATE > 0 and chg_pct >= TRAIL_ACTIVATE:
            # naive trailing using high watermark persisted in positions lot
            high_key = "_high"
            hi = max(avg_cost, max(l.get(high_key, avg_cost) for l in lots))
            if px > hi:
                # update high watermark
                for l in lots:
                    l[high_key] = px
                _save_json(POSITIONS_FILE, positions)
            pullback = (hi - px) / hi * 100.0
            if pullback >= TRAIL_DELTA:
                ok, usd, _ = place_sell(sym, qty, f"TRAIL_STOP {TRAIL_DELTA:.2f}% from peak")
                realized += usd
                continue

        # Stop-loss
        if STOP_LOSS_PCT > 0 and chg_pct <= -abs(STOP_LOSS_PCT):
            ok, usd, _ = place_sell(sym, qty, f"STOP_LOSS {STOP_LOSS_PCT:.2f}%")
            realized += usd
            continue

    if realized > 0:
        print(f"REALIZED: freed ~${realized:.2f} from sells")
    return realized

# =========================
# Buy engine (with guards)
# =========================
def can_buy(sym, price, free_usd):
    # cool-down guard
    if cooldown_active(sym):
        return False, "cooldown"

    # total positions guard
    if total_positions_count() >= MAX_POSITIONS:
        return False, "max_positions"

    # per-asset exposure guard
    exposure = position_usd_exposure(sym, price)
    if exposure + PER_TRADE_USD > PER_ASSET_CAP:
        return False, "per_asset_cap"

    # daily cap guard (comes from workflow persistence)
    if DAILY_CAP_USD <= 0:
        return False, "daily_cap_zero"

    # simple dip gate: REQUESTED DROP_PCT below recent price
    # (placeholder: if price is available we just allow; real signal is in your prior code)
    return True, "ok"

def run_cycle():
    print("=== START TRADING OUTPUT ===")
    print(f"Python {sys.version.split()[0]}")
    # 1) sells first
    evaluate_sells()

    # 2) discover universe
    uni = autopick_universe(AUTO_UNIV_COUNT)

    # 3) free USD & budget line for workflow guard
    free_usd = get_free_usd()
    print(f"FREE_USD: ${free_usd:.2f}")

    # budget: respect remaining daily cap
    budget = min(DAILY_CAP_USD, free_usd)
    buys_this_run = 0.0

    # 4) loop over universe, attempt small buys until budget exhausted
    for sym in uni:
        if budget < PER_TRADE_USD: break
        try:
            price = market_price(sym)
        except Exception:
            continue
        ok, why = can_buy(sym, price, free_usd)
        if not ok:
            # uncomment for verbose: print(f"SKIP {sym} ({why})")
            continue

        success, qty, _ = place_buy(sym, PER_TRADE_USD)
        if success:
            buys_this_run += PER_TRADE_USD
            budget -= PER_TRADE_USD

    print(f"Budget left after buys: ${budget:.2f}")
    print("=== END TRADING OUTPUT ===")
    return buys_this_run

if __name__ == "__main__":
    try:
        run_cycle()
    except KeyboardInterrupt:
        pass
