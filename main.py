# main.py — Kraken Spot: Screener + Rotation + SL/TP/Trailing
# - Buys from .state/top_candidates.json (workflow screener)
# - Rotates worst->better when full if edge >= EDGE_DELTA_PCT
# - Protects positions with STOP_LOSS, TRAIL, TAKE_PROFIT( partial )
# - Tracks positions across runs in .state/positions.json
#
# ENV (strings):
#   DRY_RUN: "ON" | "OFF" (default ON)
#   MAX_POSITIONS: "6"
#   DUST_MIN_USD: "2"
#   MAX_BUYS_PER_RUN: "1"
#   ROTATE_WHEN_CASH_SHORT: "true"
#   ROTATE_WHEN_FULL: "true"
#   EDGE_DELTA_PCT: "5"       # candidate 24h% must beat worst by >= this
#   MIN_TRADE_USD: "8"        # min USD per order
#   ALLOC_USD_PER_TRADE: ""   # optional fixed USD per buy
#   SPREAD_MAX_PCT: "0.60"    # filter wide spreads
#   MIN_NOTIONAL_USD: "5"     # skip symbols if exchange min cost higher
#
#   # --- Protection knobs ---
#   SL_PCT: "0.04"            # STOP_LOSS: sell all if price <= entry*(1-SL_PCT)
#   TRAIL_PCT: "0.035"        # TRAIL: sell all if price <= high*(1-TRAIL_PCT)
#   TP1_PCT: "0.05"           # TAKE_PROFIT 1: trim if price >= entry*(1+TP1_PCT)
#   TP1_SIZE: "0.25"          # fraction to sell on TP1 (0..1)
#
# Required secret for live orders:
#   KRAKEN_API_KEY, KRAKEN_API_SECRET

from __future__ import annotations
import os, json, time, math, csv
from pathlib import Path
from datetime import datetime, timezone

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

# ---------- small utils ---------- #

def as_bool(s: str | None, default: bool=False) -> bool:
    if s is None: return default
    s = s.strip().lower()
    return s in ("1","true","t","yes","y","on")

def as_float(s: str | None, default: float) -> float:
    try:
        if s is None or s == "": return default
        return float(s)
    except Exception:
        return default

def as_int(s: str | None, default: int) -> int:
    try:
        if s is None or s == "": return default
        return int(float(s))
    except Exception:
        return default

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")

STATE = Path(".state"); STATE.mkdir(exist_ok=True)

def green(s): return f"\x1b[32m{s}\x1b[0m"
def yellow(s): return f"\x1b[33m{s}\x1b[0m"
def red(s): return f"\x1b[31m{s}\x1b[0m"
def cyan(s): return f"\x1b[36m{s}\x1b[0m"

# ---------- env ---------- #

DRY_RUN           = as_bool(os.getenv("DRY_RUN","ON"), True)
MAX_POSITIONS     = as_int(os.getenv("MAX_POSITIONS","6"), 6)
DUST_MIN_USD      = as_float(os.getenv("DUST_MIN_USD","2"), 2.0)
MAX_BUYS_PER_RUN  = as_int(os.getenv("MAX_BUYS_PER_RUN","1"), 1)
ROTATE_WHEN_CASH_SHORT = as_bool(os.getenv("ROTATE_WHEN_CASH_SHORT","true"), True)
ROTATE_WHEN_FULL  = as_bool(os.getenv("ROTATE_WHEN_FULL","true"), True)
EDGE_DELTA_PCT    = as_float(os.getenv("EDGE_DELTA_PCT","5"), 5.0)
MIN_TRADE_USD     = as_float(os.getenv("MIN_TRADE_USD","8"), 8.0)
ALLOC_USD_PER_TRADE = os.getenv("ALLOC_USD_PER_TRADE","").strip()

SPREAD_MAX_PCT    = as_float(os.getenv("SPREAD_MAX_PCT","0.60"), 0.60)
MIN_NOTIONAL_USD  = as_float(os.getenv("MIN_NOTIONAL_USD","5"), 5.0)

# Protection
SL_PCT            = as_float(os.getenv("SL_PCT","0.04"), 0.04)         # STOP_LOSS
TRAIL_PCT         = as_float(os.getenv("TRAIL_PCT","0.035"), 0.035)    # TRAIL
TP1_PCT           = as_float(os.getenv("TP1_PCT","0.05"), 0.05)        # TAKE_PROFIT 1
TP1_SIZE          = as_float(os.getenv("TP1_SIZE","0.25"), 0.25)       # portion to sell

API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")

# ---------- exchange ---------- #

def connect_exchange():
    cfg = dict(enableRateLimit=True, timeout=20000)
    if API_KEY and API_SECRET and not DRY_RUN:
        cfg.update(apiKey=API_KEY, secret=API_SECRET)
    ex = ccxt.kraken(cfg)
    ex.load_markets()
    return ex

def map_base_to_usd_symbol(ex, base: str) -> str | None:
    base_norm = base.upper().replace("XBT","BTC")
    if base_norm in ("USD","USDT","USDC"): return None
    # prefer XBT/USD on Kraken if BTC
    candidates = ["XBT/USD"] if base_norm == "BTC" else []
    candidates.append(f"{base_norm}/USD")
    for s in candidates:
        if s in ex.markets:
            return s
    return None

def fetch_ticker_safe(ex, symbol: str) -> dict:
    try:
        return ex.fetch_ticker(symbol)
    except Exception:
        return {}

def pct24_from_ticker(t: dict) -> float | None:
    if not t: return None
    v = t.get("percentage")
    if isinstance(v,(int,float)): return float(v)
    last, open_ = t.get("last"), t.get("open")
    if isinstance(last,(int,float)) and isinstance(open_,(int,float)) and open_:
        return (last/open_ - 1.0)*100.0
    return None

def spread_pct_from_ticker(t: dict) -> float | None:
    bid, ask = t.get("bid"), t.get("ask")
    if isinstance(bid,(int,float)) and isinstance(ask,(int,float)) and bid>0 and ask>0:
        mid = (bid+ask)/2.0
        if mid>0: return (ask-bid)/mid*100.0
    return None

def last_price(t: dict) -> float | None:
    v = t.get("last")
    return float(v) if isinstance(v,(int,float)) else None

# ---------- screener ---------- #

def load_screener_or_compute(ex) -> list[dict]:
    fp = STATE/"top_candidates.json"
    if fp.exists():
        try:
            return json.loads(fp.read_text())
        except Exception:
            pass
    # quick fallback compute (simple)
    rows = []
    skip_bases = {"USDT","USDC","USD","EUR","GBP"}
    for m in ex.markets.values():
        if not m.get("spot"): continue
        if m.get("quote") != "USD": continue
        base = m.get("base","")
        if base in skip_bases: continue
        sym = m["symbol"]
        t = fetch_ticker_safe(ex, sym)
        rows.append({
            "symbol": sym,
            "pct": pct24_from_ticker(t) or -9999,
            "last": last_price(t),
            "spread": spread_pct_from_ticker(t),
            "qvol": t.get("quoteVolume") or 0.0
        })
    rows.sort(key=lambda r: r["pct"], reverse=True)
    top = rows[:12]
    fp.write_text(json.dumps(top, indent=2))
    return top

# ---------- portfolio snapshot ---------- #

def portfolio_snapshot(ex) -> dict:
    bal = ex.fetch_balance()
    totals = bal.get("total") or {}
    frees  = bal.get("free")  or {}
    cash_usd = float(frees.get("USD") or 0.0)
    holds = []
    for base, qty in totals.items():
        if not qty: continue
        b = str(base).upper()
        if b in ("USD","USDT","USDC"):  # treat stable as cash
            continue
        sym = map_base_to_usd_symbol(ex, b)
        if not sym: continue
        t = fetch_ticker_safe(ex, sym)
        p = last_price(t) or 0.0
        usd = p * float(qty)
        if usd <= 0: continue
        holds.append({
            "base": b, "symbol": sym, "qty": float(qty),
            "usd_value": float(usd),
            "price": p,
            "pct24": pct24_from_ticker(t) if pct24_from_ticker(t) is not None else -9999.0
        })
    holds.sort(key=lambda r: r["usd_value"], reverse=True)
    return {"cash_usd": cash_usd, "holds": holds}

# ---------- positions state ---------- #

POS_FILE = STATE/"positions.json"

def load_positions() -> dict:
    if POS_FILE.exists():
        try:
            return json.loads(POS_FILE.read_text())
        except Exception:
            pass
    return {}

def save_positions(d: dict):
    POS_FILE.write_text(json.dumps(d, indent=2))

def pos_init_if_missing(positions: dict, sym: str, qty: float, price: float):
    if sym not in positions:
        positions[sym] = {"entry": price, "high": price, "qty": qty, "tp1_done": False, "updated": now_iso()}

def pos_apply_buy(positions: dict, sym: str, buy_qty: float, buy_price: float):
    p = positions.get(sym)
    if not p:
        positions[sym] = {"entry": buy_price, "high": buy_price, "qty": buy_qty, "tp1_done": False, "updated": now_iso()}
        return
    # new avg entry
    new_qty = p["qty"] + buy_qty
    if new_qty <= 0:
        positions.pop(sym, None); return
    new_entry = (p["entry"]*p["qty"] + buy_price*buy_qty) / new_qty
    p["entry"] = float(new_entry)
    p["qty"] = float(new_qty)
    p["high"] = float(max(p["high"], buy_price))
    p["tp1_done"] = p.get("tp1_done", False) and (buy_price < p["entry"])  # reset if we averaged down
    p["updated"] = now_iso()

def pos_apply_sell(positions: dict, sym: str, sell_qty: float):
    p = positions.get(sym)
    if not p: return
    rem = p["qty"] - sell_qty
    if rem <= 1e-12:
        positions.pop(sym, None)
    else:
        p["qty"] = float(rem)
        p["updated"] = now_iso()

# ---------- trading primitives ---------- #

def ensure_min_cost(ex, symbol: str, usd_amount: float) -> float:
    m = ex.markets.get(symbol, {})
    limits = m.get("limits", {})
    cost = (limits.get("cost") or {}).get("min")
    if isinstance(cost,(int,float)) and cost:
        return max(float(usd_amount), float(cost))
    return usd_amount

def market_buy_usd(ex, symbol: str, usd_amount: float, dry: bool=True):
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
        return {"id":"DRY-BUY","symbol":symbol,"cost":usd_amount,"amount":amount,"price":price}
    o = ex.create_market_buy_order(symbol, amount)
    print(green(f"BUY  {symbol}  cost≈${usd_amount:.2f} amount≈{amount} -> {o.get('id')}"))
    return {"id":o.get("id"),"symbol":symbol,"amount":amount,"price":price}

def market_sell_all(ex, symbol: str, qty: float, reason: str, dry: bool=True):
    qty = float(ex.amount_to_precision(symbol, qty))
    if qty <= 0:
        raise RuntimeError(f"Sell qty too small for {symbol}")
    tag = reason.upper()
    if dry:
        print(yellow(f"DRY SELL {symbol}  qty≈{qty}  reason={tag}"))
        return {"id":"DRY-SELL","symbol":symbol,"amount":qty,"reason":tag}
    o = ex.create_market_sell_order(symbol, qty)
    print(green(f"SELL {symbol}  qty≈{qty} -> {o.get('id')}  reason={tag}"))
    return {"id":o.get("id"),"symbol":symbol,"amount":qty,"reason":tag}

def market_sell_partial(ex, symbol: str, qty: float, reason: str, dry: bool=True):
    return market_sell_all(ex, symbol, qty, reason, dry)

# ---------- guards & planning ---------- #

def choose_alloc(cash_usd: float, open_slots: int) -> float:
    if ALLOC_USD_PER_TRADE:
        return max(MIN_TRADE_USD, float(ALLOC_USD_PER_TRADE))
    open_slots = max(1, open_slots)
    return max(MIN_TRADE_USD, (cash_usd * 0.98) / open_slots)

def sweep_dust(ex, snap: dict):
    swept = []
    for h in snap["holds"]:
        if h["usd_value"] < DUST_MIN_USD:
            try:
                market_sell_all(ex, h["symbol"], h["qty"], reason="DUST", dry=DRY_RUN)
                swept.append(h["symbol"])
            except Exception as e:
                print(red(f"Dust sweep fail {h['symbol']}: {e}"))
    if swept:
        print(yellow(f"Swept dust: {', '.join(swept)}"))
    return swept

def manage_protection(ex, snap: dict, positions: dict):
    """Check STOP_LOSS, TRAIL, TAKE_PROFIT across holds. Execute sells accordingly."""
    actions = []
    for h in snap["holds"]:
        sym, qty, price = h["symbol"], h["qty"], h["price"]
        # initialize/inflate state
        pos_init_if_missing(positions, sym, qty, price)
        p = positions[sym]

        # update high watermark
        if price > p["high"]:
            p["high"] = float(price)

        entry = float(p["entry"])
        high  = float(p["high"])
        tp1_done = bool(p.get("tp1_done", False))

        sl_price    = entry * (1.0 - SL_PCT)
        trail_price = high  * (1.0 - TRAIL_PCT)
        tp1_price   = entry * (1.0 + TP1_PCT)

        # Hard STOP_LOSS (sell all)
        if price <= sl_price:
            actions.append(("STOP_LOSS", sym, qty))
            continue

        # TRAIL (sell all) — only meaningful if we've made a new high above entry
        if high > entry and price <= trail_price:
            actions.append(("TRAIL", sym, qty))
            continue

        # TAKE_PROFIT 1 (partial)
        if (not tp1_done) and price >= tp1_price and TP1_SIZE > 0:
            sell_qty = float(qty * TP1_SIZE)
            actions.append(("TAKE_PROFIT", sym, sell_qty, "TP1"))
            # mark in state after we actually execute

    # Execute
    for act in actions:
        if act[0] in ("STOP_LOSS","TRAIL"):
            _, sym, qty = act
            try:
                market_sell_all(ex, sym, qty, reason=act[0], dry=DRY_RUN)   # contains the exact tokens
                pos_apply_sell(positions, sym, qty)
            except Exception as e:
                print(red(f"Sell error {sym} ({act[0]}): {e}"))
        elif act[0] == "TAKE_PROFIT":
            _, sym, qty, _ = act
            try:
                market_sell_partial(ex, sym, qty, reason="TAKE_PROFIT", dry=DRY_RUN)
                # reduce qty & mark tp1_done
                pos_apply_sell(positions, sym, qty)
                if sym in positions:
                    positions[sym]["tp1_done"] = True
            except Exception as e:
                print(red(f"TP sell error {sym}: {e}"))

def plan_rotation(ex, screener: list[dict], snap: dict):
    holds = snap["holds"]
    cash  = snap["cash_usd"]
    held_bases = {h["base"] for h in holds}

    # screen filters
    filt = []
    for r in screener:
        sym = r["symbol"]
        if sym not in ex.markets: continue
        sp = r.get("spread")
        if isinstance(sp,(int,float)) and sp > SPREAD_MAX_PCT: 
            continue
        m = ex.markets[sym]
        min_cost = ((m.get("limits") or {}).get("cost") or {}).get("min")
        if isinstance(min_cost,(int,float)) and min_cost and min_cost > MIN_NOTIONAL_USD:
            continue
        filt.append(r)

    filt.sort(key=lambda r: r.get("pct") or -9999, reverse=True)

    to_buy, to_sell, notes = [], [], []
    slots_left = max(0, MAX_POSITIONS - len(holds))
    buys_left = MAX_BUYS_PER_RUN

    # new buys if room
    if slots_left > 0 and cash >= MIN_TRADE_USD:
        alloc = choose_alloc(cash, slots_left)
        for r in filt:
            base = r["symbol"].split("/")[0].replace("XBT","BTC")
            if base in held_bases: continue
            if buys_left <= 0 or slots_left <= 0 or cash < MIN_TRADE_USD: break
            to_buy.append({"symbol": r["symbol"], "usd": alloc, "pct": r.get("pct")})
            cash -= alloc; buys_left -= 1; slots_left -= 1

    # rotation when full or cash short
    if (ROTATE_WHEN_FULL and len(holds) >= MAX_POSITIONS) or (ROTATE_WHEN_CASH_SHORT and cash < MIN_TRADE_USD):
        if filt:
            candidate = filt[0]
            cand_pct = candidate.get("pct") or -9999
            worst = None
            for h in holds:
                if worst is None or (h["pct24"] < worst["pct24"]):
                    worst = h
            if worst and cand_pct - worst["pct24"] >= EDGE_DELTA_PCT:
                alloc = choose_alloc(cash + worst["usd_value"], 1)
                to_sell.append({"symbol": worst["symbol"], "qty": worst["qty"], "pct": worst["pct24"]})
                to_buy.append({"symbol": candidate["symbol"], "usd": alloc, "pct": cand_pct})
                notes.append(f"Rotate: {candidate['symbol']} beats {worst['symbol']} by {cand_pct - worst['pct24']:.2f}% (edge≥{EDGE_DELTA_PCT}%)")

    return {"to_sell": to_sell, "to_buy": to_buy, "notes": notes}

# ---------- main ---------- #

def main():
    print(cyan("=== Crypto Live — Screener + Rotation + SL/TP/Trailing (Kraken USD) ==="))
    print(f"UTC {now_iso()}  DRY_RUN={DRY_RUN}  MAX_POSITIONS={MAX_POSITIONS}  DUST_MIN_USD={DUST_MIN_USD}")
    print(f"PROTECT: SL_PCT={SL_PCT:.3f}  TRAIL_PCT={TRAIL_PCT:.3f}  TP1_PCT={TP1_PCT:.3f}  TP1_SIZE={TP1_SIZE:.2f}")

    ex = connect_exchange()
    screener = load_screener_or_compute(ex)
    positions = load_positions()

    # snapshot + dust
    snap = portfolio_snapshot(ex)
    print(f"Cash USD: {snap['cash_usd']:.2f}")
    if snap["holds"]:
        for h in snap["holds"]:
            print(f"  HOLD {h['symbol']:9s} qty={h['qty']:.8f} ${h['usd_value']:.2f} price={h['price']:.8f} 24h%={h['pct24']:.2f}")
    else:
        print("  Holds: (none)")

    sweep_dust(ex, snap)

    # protection sells (STOP_LOSS, TRAIL, TAKE_PROFIT)
    manage_protection(ex, snap, positions)
    save_positions(positions)

    # refresh snapshot after potential sells
    snap = portfolio_snapshot(ex)

    # rotation plan
    plan = plan_rotation(ex, screener, snap)
    (STATE/"last_plan.json").write_text(json.dumps(plan, indent=2))
    for n in plan["notes"]:
        print(yellow(f"NOTE: {n}"))

    # execute planned sells (rotation)
    for s in plan["to_sell"]:
        try:
            market_sell_all(ex, s["symbol"], s["qty"], reason="ROTATE", dry=DRY_RUN)
            pos_apply_sell(positions, s["symbol"], s["qty"])
        except Exception as e:
            print(red(f"Rotate sell error {s['symbol']}: {e}"))

    # buys
    for b in plan["to_buy"]:
        try:
            res = market_buy_usd(ex, b["symbol"], b["usd"], dry=DRY_RUN)
            # update positions on buy
            buy_price = res.get("price")
            buy_amount = res.get("amount")
            if buy_price and buy_amount:
                pos_apply_buy(positions, b["symbol"], float(buy_amount), float(buy_price))
        except Exception as e:
            print(red(f"Buy error {b['symbol']}: {e}"))

    save_positions(positions)

    # KPI summary
    kpi = STATE/"kpi_history.csv"
    summary = {
        "utc": now_iso(),
        "dry_run": DRY_RUN,
        "cash_usd": snap["cash_usd"],
        "n_holds": len(snap["holds"]),
        "duration_sec": 0,  # filled below
        "sl_pct": SL_PCT, "trail_pct": TRAIL_PCT, "tp1_pct": TP1_PCT, "tp1_size": TP1_SIZE
    }
    print(green(f"SUMMARY: {json.dumps(summary, separators=(',',':'))}"))

    write_header = not kpi.exists()
    with kpi.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["utc","dry_run","cash_usd","n_holds","duration_sec","sl_pct","trail_pct","tp1_pct","tp1_size"])
        w.writerow([summary["utc"],summary["dry_run"],summary["cash_usd"],summary["n_holds"],summary["duration_sec"],summary["sl_pct"],summary["trail_pct"],summary["tp1_pct"],summary["tp1_size"]])

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(red(f"FATAL: {e}"))
        raise
