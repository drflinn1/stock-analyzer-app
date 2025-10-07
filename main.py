# main.py — Crypto Live with In-Job Screener + Rotation (Kraken spot)
# - Reads .state/top_candidates.json (written by workflow’s Screener step)
# - Buys top candidates up to MAX_POSITIONS
# - Rotates when full (sell worst hold by 24h% if candidate beats it by EDGE_DELTA_PCT)
# - Sweeps dust (< DUST_MIN_USD) into USD
# - Safe by default: DRY_RUN=ON
#
# ENV (strings; set in workflow inputs/env):
#   DRY_RUN: "ON" | "OFF"
#   MAX_POSITIONS: "6"
#   DUST_MIN_USD: "2"
#   MAX_BUYS_PER_RUN: "1"
#   ROTATE_WHEN_CASH_SHORT: "true"
#   ROTATE_WHEN_FULL: "true"
#   EDGE_DELTA_PCT: "5"           # candidate 24h% must exceed worst-hold by >= this
#   MIN_TRADE_USD: "8"            # minimum per-trade USD notional
#   ALLOC_USD_PER_TRADE: ""       # if set, fixed USD per new buy; else auto split
#   SPREAD_MAX_PCT: "0.60"        # skip wide spreads
#   MIN_NOTIONAL_USD: "5"         # skip symbols if min cost higher than this
#
# Secrets for live trading (GitHub → Settings → Secrets and variables → Actions):
#   KRAKEN_API_KEY, KRAKEN_API_SECRET
#
# Notes:
# - This file is self-contained and defensive: if the screener file is missing,
#   it will compute a quick screener itself.
# - We only trade Kraken USD spot pairs.

from __future__ import annotations
import os, json, time, math, csv, random
from pathlib import Path
from datetime import datetime, timezone

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

# ---------- helpers ---------- #

def as_bool(s: str | None, default: bool=False) -> bool:
    if s is None: return default
    s = s.strip().lower()
    return s in ("1", "true", "t", "yes", "y", "on")

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

STATE = Path(".state")
STATE.mkdir(exist_ok=True)

def log(msg: str):
    print(msg, flush=True)

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
    pairs = [f"{base_norm}/USD"]
    if base_norm == "BTC":
        pairs.insert(0, "XBT/USD")
    for s in pairs:
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
    val = t.get("percentage")
    if isinstance(val, (int,float)):
        return float(val)
    last, open_ = t.get("last"), t.get("open")
    if isinstance(last,(int,float)) and isinstance(open_,(int,float)) and open_:
        return (last/open_ - 1.0)*100.0
    return None

def spread_pct_from_ticker(t: dict) -> float | None:
    if not t: return None
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
    # quick compute fallback (simple, same filters as workflow defaults)
    log(yellow("Screener file missing; computing a quick fallback..."))
    rows = []
    skip_bases = {"USDT","USDC","USD","EUR","GBP"}
    for m in ex.markets.values():
        if not m.get("spot"): continue
        if m.get("quote") != "USD": continue
        base = m.get("base","")
        if base in skip_bases: continue
        sym = m["symbol"]
        t = fetch_ticker_safe(ex, sym)
        pct = pct24_from_ticker(t)
        sp  = spread_pct_from_ticker(t)
        qv  = t.get("quoteVolume") or 0.0
        rows.append({"symbol": sym, "pct": pct if pct is not None else -9999, "last": last_price(t), "spread": sp, "qvol": qv})
    rows.sort(key=lambda r: (r["pct"] if r["pct"] is not None else -9999), reverse=True)
    top = rows[:12]
    (STATE/"top_candidates.json").write_text(json.dumps(top, indent=2))
    return top

# ---------- portfolio + cash ---------- #

def portfolio_snapshot(ex) -> dict:
    bal = ex.fetch_balance()
    totals = bal.get("total") or {}
    frees  = bal.get("free")  or {}
    # USD bucket
    cash_usd = float(frees.get("USD") or 0.0)
    # build holdings list: non-USD assets with USD value
    holds = []
    for base, qty in totals.items():
        if not qty: continue
        base_u = str(base).upper()
        if base_u in ("USD","USDT","USDC"):  # treat stables as cash; we trade USD spot
            continue
        sym = map_base_to_usd_symbol(ex, base_u)
        if not sym: continue
        t = fetch_ticker_safe(ex, sym)
        p = last_price(t) or 0.0
        usd_val = p * float(qty)
        if usd_val <= 0: continue
        pct24 = pct24_from_ticker(t)
        holds.append({
            "base": base_u, "symbol": sym, "qty": float(qty),
            "usd_value": float(usd_val), "pct24": pct24 if pct24 is not None else -9999.0
        })
    holds.sort(key=lambda r: r["usd_value"], reverse=True)
    return {"cash_usd": cash_usd, "holds": holds}

# ---------- trading primitives ---------- #

def ensure_min_cost(ex, symbol: str, usd_amount: float) -> float:
    m = ex.markets.get(symbol, {})
    limits = m.get("limits", {})
    cost = (limits.get("cost") or {}).get("min")
    if isinstance(cost,(int,float)) and cost:
        return max(usd_amount, float(cost))
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
        raise RuntimeError(f"Amount precision too small for {symbol}")
    if dry:
        log(yellow(f"DRY BUY  {symbol}  ~${usd_amount:.2f} @ ~{price:.6f} ≈ {amount}"))
        return {"id": "DRY-BUY", "symbol": symbol, "cost": usd_amount, "amount": amount}
    else:
        o = ex.create_market_buy_order(symbol, amount)
        log(green(f"BUY  {symbol}  cost≈${usd_amount:.2f} amount≈{amount} -> {o.get('id')}"))
        return o

def market_sell_all(ex, symbol: str, qty: float, dry: bool=True):
    qty = float(ex.amount_to_precision(symbol, qty))
    if qty <= 0:
        raise RuntimeError(f"Sell qty too small for {symbol}")
    if dry:
        log(yellow(f"DRY SELL {symbol}  qty≈{qty} (market)"))
        return {"id": "DRY-SELL", "symbol": symbol, "amount": qty}
    else:
        o = ex.create_market_sell_order(symbol, qty)
        log(green(f"SELL {symbol} qty≈{qty} -> {o.get('id')}"))
        return o

# ---------- rotation engine ---------- #

def choose_alloc(cash_usd: float, open_slots: int) -> float:
    if ALLOC_USD_PER_TRADE:
        return float(ALLOC_USD_PER_TRADE)
    open_slots = max(1, open_slots)
    # equal split with a little buffer
    return max(MIN_TRADE_USD, (cash_usd * 0.98) / open_slots)

def plan_trades(ex, screener: list[dict], snap: dict) -> dict:
    holds = snap["holds"]
    cash  = snap["cash_usd"]
    held_bases = {h["base"] for h in holds}
    # filter screener by spread & min notional
    filtered = []
    for r in screener:
        sym = r["symbol"]
        if sym not in ex.markets: continue
        sp = r.get("spread")
        if isinstance(sp,(int,float)) and sp > SPREAD_MAX_PCT: 
            continue
        # skip if exchange min cost exceeds our min notional
        m = ex.markets[sym]
        min_cost = ((m.get("limits") or {}).get("cost") or {}).get("min")
        if isinstance(min_cost, (int,float)) and min_cost and min_cost > MIN_NOTIONAL_USD:
            continue
        filtered.append(r)
    # prefer not currently held
    filtered.sort(key=lambda r: r.get("pct") or -9999, reverse=True)

    to_buy = []
    to_sell = []
    notes = []
    slots_left = max(0, MAX_POSITIONS - len(holds))
    buys_left = MAX_BUYS_PER_RUN

    # 1) If we have slots, schedule fresh buys from top of screener
    if slots_left > 0 and cash >= MIN_TRADE_USD:
        alloc = choose_alloc(cash, slots_left)
        for r in filtered:
            base = r["symbol"].split("/")[0].replace("XBT","BTC")
            if base in held_bases: 
                continue
            if buys_left <= 0 or slots_left <= 0 or cash < MIN_TRADE_USD:
                break
            to_buy.append({"symbol": r["symbol"], "usd": alloc, "pct": r.get("pct")})
            cash -= alloc
            buys_left -= 1
            slots_left -= 1

    # 2) Rotation when full / cash short
    if (ROTATE_WHEN_FULL and len(holds) >= MAX_POSITIONS) or (ROTATE_WHEN_CASH_SHORT and cash < MIN_TRADE_USD):
        if filtered:
            candidate = filtered[0]
            cand_pct = candidate.get("pct") or -9999
            # find worst holding by 24h%
            worst = None
            for h in holds:
                if worst is None or (h["pct24"] < worst["pct24"]):
                    worst = h
            if worst is not None and cand_pct - worst["pct24"] >= EDGE_DELTA_PCT:
                # rotate: sell worst, buy candidate with proceeds (or fixed alloc)
                alloc = choose_alloc(cash + worst["usd_value"], 1)
                to_sell.append({"symbol": worst["symbol"], "qty": worst["qty"], "pct": worst["pct24"]})
                to_buy.append({"symbol": candidate["symbol"], "usd": alloc, "pct": cand_pct})
                notes.append(f"Rotate: {candidate['symbol']} beats {worst['symbol']} by {cand_pct - worst['pct24']:.2f}% (edge≥{EDGE_DELTA_PCT}%)")

    return {"to_sell": to_sell, "to_buy": to_buy, "notes": notes}

# ---------- dust sweep ---------- #

def sweep_dust(ex, snap: dict):
    swept = []
    for h in snap["holds"]:
        if h["usd_value"] < DUST_MIN_USD:
            try:
                market_sell_all(ex, h["symbol"], h["qty"], DRY_RUN)
                swept.append(h["symbol"])
            except Exception as e:
                log(red(f"Dust sweep fail {h['symbol']}: {e}"))
    if swept:
        log(yellow(f"Swept dust: {', '.join(swept)}"))
    return swept

# ---------- main ---------- #

def main():
    start = time.time()
    log(cyan("=== Crypto Live — Screener + Rotation (Kraken USD) ==="))
    log(f"UTC: {now_iso()}  |  DRY_RUN={DRY_RUN}  MAX_POSITIONS={MAX_POSITIONS}  DUST_MIN_USD={DUST_MIN_USD}")

    ex = connect_exchange()
    screener = load_screener_or_compute(ex)

    # Snapshot portfolio
    snap = portfolio_snapshot(ex)
    log(f"Cash USD: {snap['cash_usd']:.2f}")
    if snap["holds"]:
        log("Holds:")
        for h in snap["holds"]:
            log(f"  - {h['symbol']:9s}  qty={h['qty']:.8f}  ${h['usd_value']:.2f}  24h%={h['pct24']:.2f}")
    else:
        log("Holds: (none)")

    # Dust sweep
    sweep_dust(ex, snap)

    # Plan trades
    plan = plan_trades(ex, screener, snap)
    (STATE/"last_plan.json").write_text(json.dumps(plan, indent=2))

    for n in plan["notes"]:
        log(yellow(f"NOTE: {n}"))

    # Execute plan: sells first, then buys
    for s in plan["to_sell"]:
        try:
            market_sell_all(ex, s["symbol"], s["qty"], DRY_RUN)
        except Exception as e:
            log(red(f"Sell error {s['symbol']}: {e}"))

    for b in plan["to_buy"]:
        try:
            market_buy_usd(ex, b["symbol"], b["usd"], DRY_RUN)
        except Exception as e:
            log(red(f"Buy error {b['symbol']}: {e}"))

    dur = time.time() - start
    # KPI-ish summary
    buys  = ", ".join([f"{b['symbol']} (${b['usd']:.2f})" for b in plan["to_buy"]]) or "-"
    sells = ", ".join([f"{s['symbol']}" for s in plan["to_sell"]]) or "-"
    summary = {
        "utc": now_iso(),
        "dry_run": DRY_RUN,
        "cash_usd": snap["cash_usd"],
        "n_holds": len(snap["holds"]),
        "buys": buys,
        "sells": sells,
        "duration_sec": round(dur,2),
    }
    print(green(f"SUMMARY: {json.dumps(summary, separators=(',',':'))}"))
    # append CSV
    kpi = STATE/"kpi_history.csv"
    write_header = not kpi.exists()
    with kpi.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["utc","dry_run","cash_usd","n_holds","buys","sells","duration_sec"])
        w.writerow([summary[k] for k in ["utc","dry_run","cash_usd","n_holds","buys","sells","duration_sec"]])

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(red(f"FATAL: {e}"))
        raise
