# stock-analyzer-app/main.py
# USD-QUOTE, MAJORS-ONLY, TOP-K ROTATION
# - Whitelist: BTC/ETH/SOL/DOGE (USD pairs only)
# - Dry-run banner + simulated fills
# - Take-profit / Stop-loss / Trailing stop
# - Cooldown after sells
# - Writes .state: .keep, day_state.json, kpi_history.csv, positions.json, cooldown.json
# - Safe if secrets are missing (paper by default)

from __future__ import annotations
import os, csv, json, math, time, random
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

# --------------------- ENV --------------------- #
def getenv(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default

DRY_RUN          = getenv("DRY_RUN", "true").lower() in ("1","true","yes","on")
EXCHANGE_ID      = getenv("EXCHANGE_ID", "kraken").lower()
QUOTE            = getenv("QUOTE", "USD").upper()                  # force USD quote
WHITELIST_RAW    = getenv("WHITELIST", "BTC,ETH,SOL,DOGE")
TOP_K            = int(getenv("TOP_K", "2"))
DOLLARS_PER_TRADE= float(getenv("DOLLARS_PER_TRADE", "25"))
TP_PCT           = float(getenv("TP_PCT", "0.035"))                # 3.5% take profit
SL_PCT           = float(getenv("SL_PCT", "0.020"))                # 2.0% stop loss
TRAIL_ARM_PCT    = float(getenv("TRAIL_ARM_PCT", "0.015"))         # arm when +1.5%
TRAIL_GIVEBACK_PCT=float(getenv("TRAIL_GIVEBACK_PCT","0.010"))     # give back 1.0%
COOLDOWN_MIN     = int(getenv("COOLDOWN_MIN", "45"))               # mins after a sell
LOOKBACK_MIN     = int(getenv("LOOKBACK_MIN", "240"))              # momentum window
MIN_NOTIONAL     = float(getenv("MIN_NOTIONAL", "10"))             # exchange min guard
MAX_DAILY_ENTRIES= int(getenv("MAX_DAILY_ENTRIES", "6"))           # sanity cap

WHITELIST = [t.strip().upper() for t in WHITELIST_RAW.split(",") if t.strip()]

# --------------------- FS STATE --------------------- #
STATE_DIR = ".state"
KEEP_FILE = os.path.join(STATE_DIR, ".keep")
DAY_STATE = os.path.join(STATE_DIR, "day_state.json")
KPI_CSV   = os.path.join(STATE_DIR, "kpi_history.csv")
POSITIONS = os.path.join(STATE_DIR, "positions.json")
COOLDOWN  = os.path.join(STATE_DIR, "cooldown.json")

def ensure_state():
    os.makedirs(STATE_DIR, exist_ok=True)
    if not os.path.exists(KEEP_FILE):
        open(KEEP_FILE, "w").write(".")

def load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r") as f: return json.load(f)
    except Exception:
        return default

def save_json(path: str, obj: Any):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def kpi_append(row: Dict[str, Any]):
    exists = os.path.exists(KPI_CSV)
    with open(KPI_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ts","mode","universe","topk","entries","exits","pnl","cash_est"
        ])
        if not exists: w.writeheader()
        w.writerow(row)

# --------------------- BROKER --------------------- #
def make_exchange() -> ccxt.Exchange:
    # Public-only by default; if secrets exist, authenticated calls will work.
    klass = getattr(ccxt, EXCHANGE_ID)
    ex = klass({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"adjustForTimeDifference": True}
    })
    key = os.environ.get("CCXT_API_KEY")
    sec = os.environ.get("CCXT_API_SECRET")
    pwd = os.environ.get("CCXT_API_PASSWORD")
    if key and sec:
        ex.apiKey = key
        ex.secret = sec
        if pwd: ex.password = pwd
    return ex

def quote_filter(symbol: str) -> bool:
    # enforce strict '/USD' quote (Kraken returns like 'BTC/USD')
    return symbol.endswith(f"/{QUOTE}")

# --------------------- SIGNALS --------------------- #
def pct(a: float, b: float) -> float:
    return (a/b - 1.0) if b else 0.0

def momentum_score(bars: List[List[float]]) -> float:
    # bars: [ts, open, high, low, close, vol]
    if len(bars) < 2: return -9e9
    c0 = bars[0][4]; c1 = bars[-1][4]
    return pct(c1, c0)

def pick_universe(ex: ccxt.Exchange) -> List[str]:
    markets = ex.load_markets()
    # Strict USD pairs only, majors whitelist.
    candidates = []
    for base in WHITELIST:
        sym = f"{base}/{QUOTE}"
        if sym in markets:
            candidates.append(sym)
    return candidates

def scan_topk(ex: ccxt.Exchange, symbols: List[str]) -> List[Tuple[str,float]]:
    scores = []
    now = ex.milliseconds()
    since = now - LOOKBACK_MIN * 60_000
    for s in symbols:
        try:
            # 5m bars gives ~LOOKBACK_MIN/5 candles
            bars = ex.fetch_ohlcv(s, timeframe="5m", since=since, limit=LOOKBACK_MIN//5 + 10)
            sc = momentum_score(bars)
            scores.append((s, sc))
        except Exception as e:
            print(f"[WARN] fetch_ohlcv failed for {s}: {e}")
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:TOP_K]

# --------------------- POSITION ENGINE --------------------- #
def now_utc_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

def load_positions() -> Dict[str, Any]:
    return load_json(POSITIONS, {})

def save_positions(p: Dict[str, Any]):
    save_json(POSITIONS, p)

def load_cooldown() -> Dict[str, float]:
    return load_json(COOLDOWN, {})

def save_cooldown(c: Dict[str, float]):
    save_json(COOLDOWN, c)

def in_cooldown(sym: str, ctab: Dict[str, float]) -> bool:
    t = ctab.get(sym)
    if not t: return False
    return (time.time() - t) < COOLDOWN_MIN*60

def arm_trailing(pos: Dict[str, Any], price: float):
    # Arm trailing when profit passes TRAIL_ARM_PCT
    if pos.get("trail_armed"): return
    if price >= pos["entry"] * (1 + TRAIL_ARM_PCT):
        pos["trail_armed"] = True
        pos["trail_peak"] = price

def update_trailing(pos: Dict[str, Any], price: float) -> bool:
    # returns True if should exit due to trailing giveback
    if not pos.get("trail_armed"): return False
    pos["trail_peak"] = max(pos.get("trail_peak", price), price)
    trigger = pos["trail_peak"] * (1 - TRAIL_GIVEBACK_PCT)
    return price <= trigger

def should_take_profit(pos: Dict[str, Any], price: float) -> bool:
    return price >= pos["entry"] * (1 + TP_PCT)

def should_stop_loss(pos: Dict[str, Any], price: float) -> bool:
    return price <= pos["entry"] * (1 - SL_PCT)

def est_qty(notional: float, price: float) -> float:
    if price <= 0: return 0.0
    # round to 6 decimals for most majors
    return max(0.0, round(notional / price, 6))

def market_price(ex: ccxt.Exchange, symbol: str) -> float:
    # ticker last or close
    t = ex.fetch_ticker(symbol)
    return float(t.get("last") or t.get("close") or 0.0)

def place_order_or_sim(ex: ccxt.Exchange, side: str, symbol: str, qty: float, price: float) -> Dict[str, Any]:
    if qty <= 0: raise ValueError("qty<=0")
    if DRY_RUN:
        print(f"SIM {side.upper()} {symbol} qty={qty} @ {price:.2f}")
        return {"id": f"sim-{int(time.time()*1000)}", "symbol": symbol, "side": side, "price": price, "filled": qty}
    else:
        # live market order (be careful!)
        try:
            if side == "buy":
                o = ex.create_market_buy_order(symbol, qty)
            else:
                o = ex.create_market_sell_order(symbol, qty)
            return o
        except Exception as e:
            print(f"[ERROR] live order failed: {e}")
            raise

# --------------------- MAIN LOOP (one pass) --------------------- #
def main():
    ensure_state()
    ex = make_exchange()

    if DRY_RUN:
        print("🚧 DRY RUN — NO REAL ORDERS SENT 🚧")

    print(f"broker={EXCHANGE_ID} quote={QUOTE} whitelist={','.join(WHITELIST)} topk={TOP_K}")

    universe = [s for s in pick_universe(ex) if quote_filter(s)]
    positions = load_positions()
    cooldown = load_cooldown()

    # --- exits first ---
    exits = 0; entries = 0; pnl = 0.0
    for sym, pos in list(positions.items()):
        if sym not in universe:
            print(f"[WARN] dropping unknown symbol from positions: {sym}")
            del positions[sym]; continue

        price = market_price(ex, sym)
        # trailing logic
        arm_trailing(pos, price)
        trail_exit = update_trailing(pos, price)
        tp = should_take_profit(pos, price)
        sl = should_stop_loss(pos, price)
        reason = None
        if trail_exit: reason = "TRAIL"
        elif tp: reason = "TP"
        elif sl: reason = "SL"

        if reason:
            qty = float(pos["qty"])
            o = place_order_or_sim(ex, "sell", sym, qty, price)
            pnl += (price - pos["entry"]) * qty
            exits += 1
            cooldown[sym] = time.time()
            print(f"EXIT {sym} via {reason}: qty={qty} @ {price:.2f} PnL={(price-pos['entry'])*qty:.2f}")
            del positions[sym]

    # --- entries: scan momentum, avoid cooldown, respect caps ---
    # crude daily entry throttle
    day = load_json(DAY_STATE, {})
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if day.get("date") != today:
        day = {"date": today, "entries": 0}
    remaining_entries = max(0, MAX_DAILY_ENTRIES - int(day.get("entries", 0)))

    picks = [s for s,_ in scan_topk(ex, universe)]
    for sym in picks:
        if remaining_entries <= 0: break
        if sym in positions: continue
        if in_cooldown(sym, cooldown): 
            print(f"SKIP {sym} (cooldown)")
            continue

        price = market_price(ex, sym)
        if DOLLARS_PER_TRADE < MIN_NOTIONAL:
            print(f"[WARN] notional ${DOLLARS_PER_TRADE:.2f} < MIN_NOTIONAL ${MIN_NOTIONAL:.2f}; skipping entries")
            break
        qty = est_qty(DOLLARS_PER_TRADE, price)
        if qty <= 0: 
            print(f"[WARN] zero qty for {sym} @ {price}")
            continue

        place_order_or_sim(ex, "buy", sym, qty, price)
        positions[sym] = {
            "entry": price,
            "qty": qty,
            "ts": now_utc_iso(),
            "trail_armed": False,
            "trail_peak": price
        }
        entries += 1
        remaining_entries -= 1
        day["entries"] = int(day.get("entries", 0)) + 1

    # persist
    save_positions(positions)
    save_cooldown(cooldown)
    save_json(DAY_STATE, day)

    # KPI
    kpi = {
        "ts": now_utc_iso(),
        "mode": "DRY" if DRY_RUN else "LIVE",
        "universe": ",".join(universe),
        "topk": len(picks),
        "entries": entries,
        "exits": exits,
        "pnl": round(pnl, 2),
        "cash_est": ""  # could fetch balances if authenticated
    }
    kpi_append(kpi)

    # summary
    green = "\033[92m"; yellow="\033[93m"; red="\033[91m"; reset="\033[0m"
    color = green if pnl > 0 or (entries>0 and exits==0) else (yellow if pnl==0 else red)
    print(color + f"KPI SUMMARY | mode={kpi['mode']} entries={entries} exits={exits} pnl={pnl:.2f} topk={kpi['topk']}" + reset)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Unhandled: {e}")
        raise
