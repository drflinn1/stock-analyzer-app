# main.py — Kraken Spot: Screener + Rotation + SL/TP/Trailing
# DRY-RUN FRIENDLY WITHOUT API KEYS (uses SIM_CASH_USD)
#
# ENV:
#   DRY_RUN: "ON" | "OFF" (default ON)
#   MAX_POSITIONS, DUST_MIN_USD, MAX_BUYS_PER_RUN, EDGE_DELTA_PCT,
#   MIN_TRADE_USD, ALLOC_USD_PER_TRADE, SPREAD_MAX_PCT, MIN_NOTIONAL_USD,
#   SL_PCT, TRAIL_PCT, TP1_PCT, TP1_SIZE
#   SIM_CASH_USD: simulated USD for DRY_RUN when no API keys (default 50)
# SECRETS (for live): KRAKEN_API_KEY, KRAKEN_API_SECRET

from __future__ import annotations
import os, json, time, math, csv
from pathlib import Path
from datetime import datetime, timezone

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

# ---------- utils ----------
def as_bool(s, default=False):
    if s is None:
        return default
    return str(s).strip().lower() in ("1", "true", "t", "yes", "y", "on")

def as_float(s, default):
    try:
        return float(s) if s not in (None, "") else default
    except:
        return default

def as_int(s, default):
    try:
        return int(float(s)) if s not in (None, "") else default
    except:
        return default

def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")

STATE = Path(".state")
STATE.mkdir(exist_ok=True)

def green(s): return f"\x1b[32m{s}\x1b[0m"
def yellow(s): return f"\x1b[33m{s}\x1b[0m"
def red(s): return f"\x1b[31m{s}\x1b[0m"
def cyan(s): return f"\x1b[36m{s}\x1b[0m"

# ---------- env ----------
DRY_RUN           = as_bool(os.getenv("DRY_RUN","ON"), True)
MAX_POSITIONS     = as_int(os.getenv("MAX_POSITIONS","6"), 6)
DUST_MIN_USD      = as_float(os.getenv("DUST_MIN_USD","2"), 2.0)
MAX_BUYS_PER_RUN  = as_int(os.getenv("MAX_BUYS_PER_RUN","1"), 1)
EDGE_DELTA_PCT    = as_float(os.getenv("EDGE_DELTA_PCT","5"), 5.0)
MIN_TRADE_USD     = as_float(os.getenv("MIN_TRADE_USD","8"), 8.0)
ALLOC_USD_PER_TRADE = os.getenv("ALLOC_USD_PER_TRADE","").strip()
SPREAD_MAX_PCT    = as_float(os.getenv("SPREAD_MAX_PCT","0.60"), 0.60)
MIN_NOTIONAL_USD  = as_float(os.getenv("MIN_NOTIONAL_USD","5"), 5.0)

SL_PCT            = as_float(os.getenv("SL_PCT","0.04"), 0.04)
TRAIL_PCT         = as_float(os.getenv("TRAIL_PCT","0.035"), 0.035)
TP1_PCT           = as_float(os.getenv("TP1_PCT","0.05"), 0.05)
TP1_SIZE          = as_float(os.getenv("TP1_SIZE","0.25"), 0.25)

SIM_CASH_USD      = as_float(os.getenv("SIM_CASH_USD","50"), 50.0)

API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")
HAS_KEYS = bool(API_KEY and API_SECRET)

# ---------- exchange ----------
def connect_exchange():
    cfg = dict(enableRateLimit=True, timeout=20000)
    if HAS_KEYS and not DRY_RUN:
        cfg.update(apiKey=API_KEY, secret=API_SECRET)
    ex = ccxt.kraken(cfg)
    ex.load_markets()
    return ex

def map_base_to_usd_symbol(ex, base: str) -> str | None:
    base = base.upper().replace("XBT", "BTC")
    if base in ("USD", "USDT", "USDC"):
        return None
    candidates = []
    if base == "BTC":
        candidates.append("XBT/USD")
    candidates.append(f"{base}/USD")
    for s in candidates:
        if s in ex.markets:
            return s
    return None

def fetch_ticker_safe(ex, symbol):
    try:
        return ex.fetch_ticker(symbol)
    except Exception:
        return {}

def pct24(t):
    v = t.get("percentage")
    if isinstance(v, (int, float)):
        return float(v)
    last, open_ = t.get("last"), t.get("open")
    if isinstance(last, (int, float)) and isinstance(open_, (int, float)) and open_:
        return (last/open_ - 1) * 100.0
    return None

def spread_pct(t):
    bid, ask = t.get("bid"), t.get("ask")
    if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and bid > 0 and ask > 0:
        mid = (bid + ask) / 2
        if mid > 0:
            return (ask - bid) / mid * 100.0
    return None

def last_price(t):
    v = t.get("last")
    return float(v) if isinstance(v, (int, float)) else None

# ---------- screener ----------
def load_screener_or_compute(ex):
    fp = STATE / "top_candidates.json"
    if fp.exists():
        try:
            return json.loads(fp.read_text())
        except Exception:
            pass

    # fallback compute quickly
    rows = []
    skip = {"USDT", "USDC", "USD", "EUR", "GBP"}
    for m in ex.markets.values():
        if not m.get("spot"):
            continue
        if m.get("quote") != "USD":
            continue
        base = m.get("base", "")
        if base in skip:
            continue
        sym = m["symbol"]
        t = fetch_ticker_safe(ex, sym)
        rows.append({
            "symbol": sym,
            "pct": pct24(t) or -9999,
            "last": last_price(t),
            "spread": spread_pct(t),
            "qvol": t.get("quoteVolume") or 0.0
        })
    rows.sort(key=lambda r: r["pct"], reverse=True)
    top = rows[:12]
    fp.write_text(json.dumps(top, indent=2))
    return top

# ---------- portfolio ----------
def portfolio_snapshot(ex):
    """If we have keys (or live), read real balance. Otherwise simulate in DRY_RUN."""
    if HAS_KEYS:
        bal = ex.fetch_balance()
        totals = bal.get("total") or {}
        frees  = bal.get("free")  or {}
        cash_usd = float(frees.get("USD") or 0.0)
        holds = []
        for base, qty in totals.items():
            if not qty:
                continue
            b = str(base).upper()
            if b in ("USD", "USDT", "USDC"):
                continue
            sym = map_base_to_usd_symbol(ex, b)
            if not sym:
                continue
            t = fetch_ticker_safe(ex, sym)
            p = last_price(t) or 0.0
            usd = p * float(qty)
            if usd <= 0:
                continue
            holds.append({
                "base": b, "symbol": sym, "qty": float(qty),
                "usd_value": float(usd), "price": p,
                "pct24": pct24(t) if pct24(t) is not None else -9999.0
            })
        holds.sort(key=lambda r: r["usd_value"], reverse=True)
        return {"cash_usd": cash_usd, "holds": holds}
    else:
        # Simulated portfolio for DRY RUN without keys
        return {"cash_usd": SIM_CASH_USD, "holds": []}

# ---------- position state ----------
POS_FILE = STATE / "positions.json"

def load_positions():
    if POS_FILE.exists():
        try:
            return json.loads(POS_FILE.read_text())
        except Exception:
            pass
    return {}

def save_positions(d):
    POS_FILE.write_text(json.dumps(d, indent=2))

def pos_init_if_missing(positions, sym, qty, price):
    if sym not in positions:
        positions[sym] = {"entry": price, "high": price, "qty": qty, "tp1_done": False, "updated": now_iso()}

def pos_apply_buy(positions, sym, buy_qty, buy_price):
    p = positions.get(sym)
    if not p:
        positions[sym] = {"entry": buy_price, "high": buy_price, "qty": buy_qty, "tp1_done": False, "updated": now_iso()}
        return
    new_qty = p["qty"] + buy_qty
    if new_qty <= 0:
        positions.pop(sym, None)
        return
    p["entry"] = (p["entry"]*p["qty"] + buy_price*buy_qty) / new_qty
    p["qty"]   = new_qty
    p["high"]  = max(p["high"], buy_price)
    if buy_price < p["entry"]:
        p["tp1_done"] = False
    p["updated"] = now_iso()

def pos_apply_sell(positions, sym, sell_qty):
    p = positions.get(sym)
    if not p:
        return
    rem = p["qty"] - sell_qty
    if rem <= 1e-12:
        positions.pop(sym, None)
    else:
        p["qty"] = rem
        p["updated"] = now_iso()

# ---------- order helpers ----------
def ensure_min_cost(ex, symbol, usd_amount):
    m = ex.markets.get(symbol, {})
    cost = ((m.get("limits") or {}).get("cost") or {}).get("min")
    if isinstance(cost, (int, float)) and cost:
        return max(float(usd_amount), float(cost))
    return usd_amount

def market_buy_usd(ex, symbol, usd_amount, dry=True):
    usd_amount = ensure_min_cost(ex, symbol, usd_amount)
    t = fetch_ticker_safe(ex, symbol)
    price = last_price(t)
    if not price or price <= 0:
        raise RuntimeError(f"No price for {symbol}")
    amount = usd_amount / price
    amount = float(ex.amount_to_precision(symbol, amount))
    if amount <= 0:
        raise RuntimeError(f"Amount too small for {symbol}")
    if dry:
        print(yellow(f"DRY BUY  {symbol}  ~${usd_amount:.2f} @ ~{price:.8f} ≈ {amount}"))
        return {"id":"DRY-BUY","symbol":symbol,"amount":amount,"price":price}
    o = ex.create_market_buy_order(symbol, amount)
    print(green(f"BUY  {symbol}  amount≈{amount} -> {o.get('id')}"))
    return {"id":o.get("id"),"symbol":symbol,"amount":amount,"price":price}

def market_sell_all(ex, symbol, qty, reason, dry=True):
    qty = float(ex.amount_to_precision(symbol, qty))
    if qty <= 0:
        raise RuntimeError(f"Sell qty too small for {symbol}")
    tag = reason.upper()
    if dry:
        print(yellow(f"DRY SELL {symbol} qty≈{qty} reason={tag}"))
        return {"id":"DRY-SELL","symbol":symbol,"amount":qty,"reason":tag}
    o = ex.create_market_sell_order(symbol, qty)
    print(green(f"SELL {symbol} qty≈{qty} -> {o.get('id')} reason={tag}"))
    return {"id":o.get("id"),"symbol":symbol,"amount":qty,"reason":tag}

# ---------- guards ----------
def choose_alloc(cash_usd, open_slots):
    if ALLOC_USD_PER_TRADE:
        return max(MIN_TRADE_USD, float(ALLOC_USD_PER_TRADE))
    open_slots = max(1, open_slots)
    return max(MIN_TRADE_USD, (cash_usd * 0.98) / open_slots)

def sweep_dust(ex, snap):
    if not HAS_KEYS:
        return []
    swept = []
    for h in snap["holds"]:
        if h["usd_value"] < DUST_MIN_USD:
            try:
                market_sell_all(ex, h["symbol"], h["qty"], "DUST", DRY_RUN)
                swept.append(h["symbol"])
            except Exception as e:
                print(red(f"Dust sweep fail {h['symbol']}: {e}"))
    if swept:
        print(yellow(f"Swept dust: {', '.join(swept)}"))
    return swept

def manage_protection(ex, snap, positions):
    if not HAS_KEYS:
        return
    actions = []
    for h in snap["holds"]:
        sym, qty, price = h["symbol"], h["qty"], h["price"]
        pos_init_if_missing(positions, sym, qty, price)
        p = positions[sym]
        if price > p["high"]:
            p["high"] = price
        entry = p["entry"]
        high  = p["high"]
        tp1_done = p.get("tp1_done", False)
        sl_price    = entry * (1 - SL_PCT)
        trail_price = high  * (1 - TRAIL_PCT)
        tp1_price   = entry * (1 + TP1_PCT)
        if price <= sl_price:
            actions.append(("STOP_LOSS", sym, qty))
            continue
        if high > entry and price <= trail_price:
            actions.append(("TRAIL", sym, qty))
            continue
        if (not tp1_done) and price >= tp1_price and TP1_SIZE > 0:
            actions.append(("TAKE_PROFIT", sym, float(qty * TP1_SIZE)))
    for kind, sym, qty in actions:
        try:
            market_sell_all(ex, sym, qty, reason=kind, dry=DRY_RUN)
            pos_apply_sell(positions, sym, qty)
            if kind == "TAKE_PROFIT" and sym in positions:
                positions[sym]["tp1_done"] = True
        except Exception as e:
            print(red(f"Sell error {sym} ({kind}): {e}"))

def plan_rotation(ex, screener, snap):
    holds = snap["holds"]
    cash  = snap["cash_usd"]
    held_bases = {h["base"] for h in holds}
    filt = []
    for r in screener:
        sym = r["symbol"]
        if sym not in ex.markets:
            continue
        sp = r.get("spread")
        if isinstance(sp, (int, float)) and sp > SPREAD_MAX_PCT:
            continue
        m = ex.markets[sym]
        min_cost = ((m.get("limits") or {}).get("cost") or {}).get("min")
        if isinstance(min_cost, (int, float)) and min_cost and min_cost > MIN_NOTIONAL_USD:
            continue
        filt.append(r)
    filt.sort(key=lambda r: r.get("pct") or -9999, reverse=True)
    to_buy, to_sell, notes = [], [], []
    slots_left = max(0, MAX_POSITIONS - len(holds))
    buys_left  = MAX_BUYS_PER_RUN
    if slots_left > 0 and cash >= MIN_TRADE_USD:
        alloc = choose_alloc(cash, slots_left)
        for r in filt:
            base = r["symbol"].split("/")[0].replace("XBT", "BTC")
            if base in held_bases:
                continue
            if buys_left <= 0 or slots_left <= 0 or cash < MIN_TRADE_USD:
                break
            to_buy.append({"symbol": r["symbol"], "usd": alloc, "pct": r.get("pct")})
            cash -= alloc
            buys_left -= 1
            slots_left -= 1
    if (len(holds) >= MAX_POSITIONS) or (cash < MIN_TRADE_USD):
        if filt:
            cand = filt[0]
            cand_pct = cand.get("pct") or -9999
            worst = None
            for h in holds:
                if worst is None or h["pct24"] < worst["pct24"]:
                    worst = h
            if worst and cand_pct - worst["pct24"] >= EDGE_DELTA_PCT:
                alloc = choose_alloc(cash + worst["usd_value"], 1)
                to_sell.append({"symbol": worst["symbol"], "qty": worst["qty"], "pct": worst["pct24"]})
                to_buy.append({"symbol": cand["symbol"], "usd": alloc, "pct": cand_pct})
                notes.append(f"Rotate: {cand['symbol']} beats {worst['symbol']} by {cand_pct - worst['pct24']:.2f}% (edge≥{EDGE_DELTA_PCT}%)")
    return {"to_sell": to_sell, "to_buy": to_buy, "notes": notes}

# ---------- main ----------
def main():
    print(cyan("=== Crypto Live — Screener + Rotation + SL/TP/Trailing (Kraken USD) ==="))
    print(f"UTC {now_iso()} DRY_RUN={DRY_RUN} HAS_KEYS={HAS_KEYS} SIM_CASH_USD={SIM_CASH_USD}")
    print(f"PROTECT SL={SL_PCT:.3f} TRAIL={TRAIL_PCT:.3f} TP1={TP1_PCT:.3f} x {TP1_SIZE:.2f}")

    ex = connect_exchange()
    screener = load_screener_or_compute(ex)
    positions = load_positions()

    snap = portfolio_snapshot(ex)
    print(f"Cash USD: {snap['cash_usd']:.2f}")
    if snap["holds"]:
        for h in snap["holds"]:
            print(f"  HOLD {h['symbol']:9s} qty={h['qty']:.8f} ${h['usd_value']:.2f} price={h['price']:.8f} 24h%={h['pct24']:.2f}")
    else:
        print("  Holds: (none)")

    sweep_dust(ex, snap)
    manage_protection(ex, snap, positions)
    save_positions(positions)

    snap = portfolio_snapshot(ex)

    plan = plan_rotation(ex, screener, snap)
    (STATE/"last_plan.json").write_text(json.dumps(plan, indent=2))
    for n in plan["notes"]:
        print(yellow(f"NOTE: {n}"))

    for s in plan["to_sell"]:
        try:
            market_sell_all(ex, s["symbol"], s["qty"], "ROTATE", DRY_RUN)
            pos_apply_sell(positions, s["symbol"], s["qty"])
        except Exception as e:
            print(red(f"Rotate sell error {s['symbol']}: {e}"))

    for b in plan["to_buy"]:
        try:
            res = market_buy_usd(ex, b["symbol"], b["usd"], DRY_RUN)
            if res.get("price") and res.get("amount"):
                pos_apply_buy(positions, b["symbol"], float(res["amount"]), float(res["price"]))
        except Exception as e:
            print(red(f"Buy error {b['symbol']}: {e}"))

    save_positions(positions)

    # KPI
    kpi = STATE / "kpi_history.csv"
    write_header = not kpi.exists()
    with kpi.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["utc","dry_run","cash_usd","n_holds","sl","trail","tp1","tp1_size"])
        w.writerow([now_iso(), DRY_RUN, snap["cash_usd"], len(snap["holds"]), SL_PCT, TRAIL_PCT, TP1_PCT, TP1_SIZE])

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(red(f"FATAL: {e}"))
        raise
