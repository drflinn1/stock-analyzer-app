#!/usr/bin/env python3
import os, time, json, math, re, sys, random, datetime
from collections import defaultdict
from typing import Any, Dict, List, Optional

# =========================
# Config via environment
# =========================
EXCHANGE_NAME   = os.getenv("EXCHANGE", "kraken").lower()
MODE            = os.getenv("MODE", "live").lower()              # "live" or "dry"
DRY_RUN         = MODE != "live" or os.getenv("DRY_RUN", "false").lower() == "true"

PER_TRADE_USD   = float(os.getenv("PER_TRADE_USD", "10"))
DAILY_CAP_USD   = float(os.getenv("DAILY_CAP_USD", "25"))
DROP_PCT        = float(os.getenv("DROP_PCT", "2.0"))

TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "2.0"))
TRAIL_ACTIVATE  = float(os.getenv("TRAILING_ACTIVATE_PCT", "1.0"))
TRAIL_DELTA     = float(os.getenv("TRAILING_DELTA_PCT", "1.0"))
STOP_LOSS_PCT   = float(os.getenv("STOP_LOSS_PCT", "0"))

MAX_POSITIONS   = int(os.getenv("MAX_POSITIONS", "50"))
PER_ASSET_CAP   = float(os.getenv("PER_ASSET_CAP_USD", "50"))
COOLDOWN_MIN    = int(os.getenv("COOLDOWN_MINUTES", "30"))

AUTO_UNIV_COUNT = int(os.getenv("AUTO_UNIVERSE_COUNT", "500"))
QUOTE           = os.getenv("QUOTE", "USD").upper()

MARKET          = os.getenv("MARKET", "crypto").lower()
SELL_DEBUG      = int(os.getenv("SELL_DEBUG", "0"))

STATE_DIR       = ".state"
POSITIONS_FILE  = os.path.join(STATE_DIR, "positions.json")
COOLDOWN_FILE   = os.path.join(STATE_DIR, "cooldown.json")

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

positions: Dict[str, Any] = _load_json(POSITIONS_FILE, {})
cooldown: Dict[str, str]  = _load_json(COOLDOWN_FILE, {})

# ---------- Auto-heal ----------
def _normalize_lot(x: Any):
    if isinstance(x, dict) and "qty" in x and "cost" in x:
        try:
            return {
                "qty": float(x["qty"]),
                "cost": float(x["cost"]),
                "entry": x.get("entry") or datetime.datetime.utcnow().isoformat(),
                **{k: v for k, v in x.items() if k.startswith("_")}
            }
        except Exception:
            return None
    return None

def normalize_positions():
    changed = False
    if not isinstance(positions, dict):
        print("WARN: positions.json invalid; resetting")
        positions.clear()
        _save_json(POSITIONS_FILE, positions)
        return
    to_delete = []
    for sym, lots in list(positions.items()):
        new_lots: List[dict] = []
        if isinstance(lots, str):
            try:
                lots = json.loads(lots); changed = True
            except Exception:
                print(f"WARN: positions[{sym}] non-JSON string; dropping")
                to_delete.append(sym); continue
        if isinstance(lots, dict):
            lot = _normalize_lot(lots); 
            if lot: new_lots = [lot]; changed = True
        elif isinstance(lots, list):
            for l in lots:
                lot = _normalize_lot(l)
                if lot: new_lots.append(lot)
                else: changed = True
        else:
            print(f"WARN: positions[{sym}] unexpected type {type(lots)}; dropping")
            to_delete.append(sym); continue
        if new_lots: positions[sym] = new_lots
        else: to_delete.append(sym)
    for sym in to_delete: positions.pop(sym, None)
    if changed or to_delete:
        print("NOTE: normalized positions.json")
        _save_json(POSITIONS_FILE, positions)

normalize_positions()

# =========================
# Exchange (CCXT)
# =========================
try:
    import ccxt  # type: ignore
except Exception:
    print("ccxt not available; add to requirements.txt", file=sys.stderr)
    ccxt = None

def mk_exchange():
    if EXCHANGE_NAME == "kraken":
        ex = ccxt.kraken({
            "apiKey": os.getenv("KRAKEN_API_KEY", ""),
            "secret": os.getenv("KRAKEN_API_SECRET", ""),
            "enableRateLimit": True,
        })
    else:
        raise RuntimeError(f"Unsupported exchange: {EXCHANGE_NAME}")
    ex.load_markets()
    return ex

exchange = mk_exchange() if ccxt else None

def get_free_usd():
    if not exchange: return 0.0
    try:
        bal = exchange.fetch_free_balance()
        return float(bal.get(QUOTE, 0.0))
    except Exception:
        return 0.0

def market_price(symbol):
    if not exchange: return 0.0
    t = exchange.fetch_ticker(symbol)
    return float(t["last"] or t["close"] or 0.0)

def list_tradeable_pairs():
    if not exchange: return []
    return sorted(set(
        m["symbol"] for m in exchange.markets.values()
        if m.get("spot") and m.get("active") and m.get("quote") == QUOTE
    ))

# =========================
# Universe
# =========================
def autopick_universe(limit=AUTO_UNIV_COUNT):
    symbols = list_tradeable_pairs()
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
def _lots(sym) -> List[dict]:
    v = positions.get(sym, [])
    return v if isinstance(v, list) else []

def total_positions_count():
    return sum(1 for lots in positions.values() if isinstance(lots, list) for _ in lots)

def position_usd_exposure(sym, price):
    lots = _lots(sym)
    qty  = sum(float(l.get("qty", 0.0)) for l in lots if isinstance(l, dict))
    return qty * price

def add_lot(sym, qty, cost):
    lots = _lots(sym)
    lots.append({"qty": float(qty), "cost": float(cost), "entry": datetime.datetime.utcnow().isoformat()})
    positions[sym] = lots
    _save_json(POSITIONS_FILE, positions)

def clear_sym(sym):
    positions.pop(sym, None)
    _save_json(POSITIONS_FILE, positions)

def cooldown_active(sym):
    if sym not in cooldown: return False
    try:
        t = datetime.datetime.fromisoformat(cooldown[sym])
        return (datetime.datetime.utcnow() - t) < datetime.timedelta(minutes=COOLDOWN_MIN)
    except Exception:
        return False

def mark_sold(sym):
    cooldown[sym] = datetime.datetime.utcnow().isoformat()
    _save_json(COOLDOWN_FILE, cooldown)

# =========================
# Tax Ledger
# =========================
try:
    from tax_ledger import TaxLedger
    _ledger = TaxLedger()
    print(f"[boot] TaxLedger ready → {_ledger.ledger_path}")
except Exception as e:
    _ledger = None
    print(f"[warn] TaxLedger not available: {e}")

def _maybe_tax_log_per_lot(sym: str, lots: List[dict], sell_px: float, order_id: Optional[str] = None):
    if DRY_RUN or _ledger is None:
        if DRY_RUN and SELL_DEBUG:
            try:
                profit_preview = sum(max(0.0, (sell_px - float(l["cost"])) * float(l["qty"])) for l in lots)
                print(f"[tax][dry] preview (not recorded) potential profit ~${profit_preview:.2f}")
            except Exception:
                pass
        return
    reserved_total = 0.0
    profit_total = 0.0
    for l in lots:
        try:
            qty = float(l["qty"]); cost = float(l["cost"])
            proceeds = qty * sell_px
            cost_basis = qty * cost
            opened_at = l.get("entry")
            hp_days = None
            if isinstance(opened_at, str):
                try:
                    opened_dt = datetime.datetime.fromisoformat(opened_at.replace("Z",""))
                    hp_days = (datetime.datetime.utcnow() - opened_dt).days
                except Exception:
                    hp_days = None
            res = _ledger.record_sell(
                market=MARKET, symbol=sym, qty=qty, avg_price_usd=sell_px,
                proceeds_usd=proceeds, cost_basis_usd=cost_basis, fees_usd=0.0,
                holding_period_days=hp_days, run_id=os.getenv("GITHUB_RUN_ID"), trade_id=order_id
            )
            profit_total += float(res.get("profit_usd", 0.0))
            reserved_total += float(res.get("reserved_usd", 0.0))
        except Exception as e:
            print(f"[tax][error] lot-record failed for {sym}: {e}")
    print(f"[tax] profit=${profit_total:.2f} reserved=${reserved_total:.2f} "
          f"(rate={getattr(_ledger,'reserve_rate',0):.2f}+{getattr(_ledger,'state_rate',0):.2f}) "
          f"→ {_ledger.ledger_path}")

# =========================
# Order helpers
# =========================
def place_buy(sym, usd_amt):
    px = market_price(sym)
    if px <= 0: return False, 0.0, "bad_price"
    qty = usd_amt / px
    if DRY_RUN:
        print(f"BUY {sym}: DRY ~${usd_amt:.2f} @ ~{px:.8f} qty~{qty:.8f}")
        add_lot(sym, qty, px)
        return True, qty, "dry"
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
    lots_snapshot = [l for l in _lots(sym) if isinstance(l, dict) and "qty" in l and "cost" in l]
    if DRY_RUN:
        print(f"SELL {sym}: {reason} sold {qty:.6f} ~${usd:.2f}")
        _maybe_tax_log_per_lot(sym, lots_snapshot, px, order_id=None)
        clear_sym(sym); mark_sold(sym)
        return True, usd, "dry"
    try:
        order = exchange.create_market_sell_order(sym, qty)
        oid = order.get('id', '?')
        print(f"SELL {sym}: {reason} sold {qty:.6f} ~${usd:.2f} (order id {oid})")
        _maybe_tax_log_per_lot(sym, lots_snapshot, px, order_id=str(oid))
        clear_sym(sym); mark_sold(sym)
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
        if not isinstance(lots, list) or not lots: 
            continue
        valid = [l for l in lots if isinstance(l, dict) and "qty" in l and "cost" in l]
        if not valid: 
            continue
        qty = sum(float(l["qty"]) for l in valid)
        if qty <= 0: 
            continue
        avg_cost = sum(float(l["qty"])*float(l["cost"]) for l in valid)/qty
        px = market_price(sym)
        if px <= 0: 
            continue
        chg_pct = (px - avg_cost) / avg_cost * 100.0

        hi_key = "_high"
        hi = max([float(l.get(hi_key, avg_cost)) for l in valid] + [avg_cost])
        if px > hi:
            for l in valid: l[hi_key] = px
            positions[sym] = valid; _save_json(POSITIONS_FILE, positions)
            hi = px
        pullback = (hi - px) / hi * 100.0 if hi > 0 else 0.0

        if SELL_DEBUG:
            print(f"[sellchk] {sym} qty={qty:.6f} avg_cost={avg_cost:.8f} px={px:.8f} "
                  f"chg={chg_pct:.2f}% hi={hi:.8f} pullback={pullback:.2f}% "
                  f"TP>={TAKE_PROFIT_PCT:.2f}% TRAIL_ACT>={TRAIL_ACTIVATE:.2f}% "
                  f"DELTA>={TRAIL_DELTA:.2f}% SL<={-abs(STOP_LOSS_PCT):.2f}%")

        # Take-profit
        if TAKE_PROFIT_PCT > 0 and chg_pct >= TAKE_PROFIT_PCT:
            ok, usd, _ = place_sell(sym, qty, f"TAKE_PROFIT {TAKE_PROFIT_PCT:.2f}%")
            realized += usd if ok else 0.0
            if SELL_DEBUG: print(f"[sellchk] -> TAKE_PROFIT fired for {sym}")
            continue

        # Trailing
        if TRAIL_ACTIVATE > 0 and chg_pct >= TRAIL_ACTIVATE and pullback >= TRAIL_DELTA:
            ok, usd, _ = place_sell(sym, qty, f"TRAIL_STOP {TRAIL_DELTA:.2f}% from peak")
            realized += usd if ok else 0.0
            if SELL_DEBUG: print(f"[sellchk] -> TRAIL_STOP fired for {sym}")
            continue

        # Stop-loss
        if STOP_LOSS_PCT > 0 and chg_pct <= -abs(STOP_LOSS_PCT):
            ok, usd, _ = place_sell(sym, qty, f"STOP_LOSS {STOP_LOSS_PCT:.2f}%")
            realized += usd if ok else 0.0
            if SELL_DEBUG: print(f"[sellchk] -> STOP_LOSS fired for {sym}")
            continue

    if realized > 0:
        print(f"REALIZED: freed ~${realized:.2f} from sells")
    return realized

# =========================
# Buy engine (with guards)
# =========================
def can_buy(sym, price, free_usd):
    if cooldown_active(sym): return False, "cooldown"
    if total_positions_count() >= MAX_POSITIONS: return False, "max_positions"
    exposure = position_usd_exposure(sym, price)
    if exposure + PER_TRADE_USD > PER_ASSET_CAP: return False, "per_asset_cap"
    if DAILY_CAP_USD <= 0: return False, "daily_cap_zero"
    return True, "ok"

def run_cycle():
    print("=== START TRADING OUTPUT ===")
    print(f"Python {sys.version.split()[0]}")

    evaluate_sells()

    uni = autopick_universe(AUTO_UNIV_COUNT)

    free_usd = get_free_usd()
    print(f"FREE_USD: ${free_usd:.2f}")

    budget = min(DAILY_CAP_USD, free_usd)
    buys_this_run = 0.0

    for sym in uni:
        if budget < PER_TRADE_USD: break
        try:
            price = market_price(sym)
        except Exception:
            continue
        ok, _ = can_buy(sym, price, free_usd)
        if not ok: continue
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
