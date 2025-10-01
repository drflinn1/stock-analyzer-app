# main.py — Crypto Live with Guard Pack (root-level, FULL FILE)

from __future__ import annotations
import os, json, time, math, csv, sys
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

STATE_DIR = ".state"
POSITIONS_F = os.path.join(STATE_DIR, "positions.json")
KPI_CSV = os.path.join(STATE_DIR, "kpi_history.csv")
SUMMARY_F = os.path.join(STATE_DIR, "summary_last.txt")
os.makedirs(STATE_DIR, exist_ok=True)

def env_str(k:str, d:str="") -> str: return str(os.getenv(k, d)).strip()
def env_f(k:str, d:float) -> float:
    try: return float(env_str(k, str(d)))
    except: return d
def env_i(k:str, d:int) -> int:
    try: return int(env_str(k, str(d)))
    except: return d
def env_b(k:str, d:bool) -> bool:
    v = env_str(k, "true" if d else "false").lower()
    return v in ("1","true","yes","y")

# ---------- ENV ----------
DRY_RUN = env_b("DRY_RUN", True)
LIVE_CONFIRM = env_str("LIVE_CONFIRM", "")
# Safety: unless LIVE_CONFIRM is exactly 'I_UNDERSTAND', force dry-run.
if LIVE_CONFIRM != "I_UNDERSTAND":
    DRY_RUN = True

# Hard fallback so schedule runs don't break even if EXCHANGE_ID=""
EXCHANGE_ID = (env_str("EXCHANGE_ID", "kraken") or "kraken").lower()
MAX_ENTRIES_PER_RUN = env_i("MAX_ENTRIES_PER_RUN", 1)
USD_PER_TRADE = env_f("USD_PER_TRADE", 10.0)
TAKE_PROFIT_PCT = env_f("TAKE_PROFIT_PCT", 0.035)
STOP_LOSS_PCT   = env_f("STOP_LOSS_PCT",   0.020)
TRAIL_PCT       = env_f("TRAIL_PCT",       0.025)
DAILY_LOSS_CAP_PCT = env_f("DAILY_LOSS_CAP_PCT", -0.02)
RESERVE_USD = env_f("RESERVE_USD", 100.0)
UNIVERSE_MODE = env_str("UNIVERSE_MODE", "whitelist")
WHITELIST = [s.strip() for s in env_str("WHITELIST", "BTC/USD,ETH/USD,SOL/USD,DOGE/USD").split(",") if s.strip()]
TOPK = env_i("TOPK", 8)
MAX_POSITIONS = env_i("MAX_POSITIONS", 6)
AVOID_STABLES = env_b("AVOID_STABLES", True)

GREEN = "\033[92m"; YELLOW="\033[93m"; RED="\033[91m"; RESET="\033[0m"
def log_good(msg): print(GREEN + msg + RESET, flush=True)
def log_warn(msg): print(YELLOW + msg + RESET, flush=True)
def log_error(msg): print(RED + msg + RESET, flush=True)

def now_utc_ts() -> int: return int(time.time())
def now_iso() -> str: return datetime.now(timezone.utc).isoformat(timespec="seconds")

def load_positions() -> Dict[str, dict]:
    if not os.path.exists(POSITIONS_F): return {}
    try:
        with open(POSITIONS_F, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def save_positions(d: Dict[str, dict]) -> None:
    with open(POSITIONS_F, "w", encoding="utf-8") as f: json.dump(d, f, indent=2, sort_keys=True)

def append_kpi(row: Dict[str, str]) -> None:
    new_file = not os.path.exists(KPI_CSV)
    with open(KPI_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ts","iso","pnl_realized","pnl_unrealized","positions","buys","sells","dry_run"
        ])
        if new_file: w.writeheader()
        w.writerow(row)

def stable_like(symbol: str) -> bool:
    s = symbol.upper()
    stubs = ("USDT/","USDC/","DAI/","TUSD/","FDUSD/","EUR/","GBP/","USD/")
    return any(s.startswith(x) for x in stubs) or any(s.endswith(x) for x in ("/USDT","/USDC","/DAI","/TUSD","/FDUSD","/EUR","/GBP","/USD"))

def usd_keys() -> Tuple[str,...]:
    return ("USD","ZUSD")

def fetch_usd_free(ex) -> float:
    bal = ex.fetch_balance()
    total = 0.0
    for k in usd_keys():
        if k in bal and isinstance(bal[k], dict):
            total += float(bal[k].get("free", 0) or 0)
        elif k in bal:
            total += float(bal.get(k, 0) or 0)
    if "free" in bal and isinstance(bal["free"], dict):
        for k in usd_keys():
            total += float(bal["free"].get(k, 0) or 0)
    return total

def get_exchange():
    try:
        if EXCHANGE_ID == "kraken":
            apiKey = os.getenv("KRAKEN_API_KEY","")
            secret = os.getenv("KRAKEN_API_SECRET","")
            ex = ccxt.kraken({
                "apiKey": apiKey,
                "secret": secret,
                "enableRateLimit": True
            })
        else:
            if not hasattr(ccxt, EXCHANGE_ID):
                raise ValueError(f"Unknown EXCHANGE_ID: {EXCHANGE_ID}")
            ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
        ex.load_markets()
        return ex
    except Exception as e:
        raise SystemExit(f"exchange init failed: {e}")

def universe_auto(ex) -> List[str]:
    tickers = ex.fetch_tickers()
    scored = []
    for sym, t in tickers.items():
        if "/" not in sym: continue
        if AVOID_STABLES and (stable_like(sym)): continue
        qv = float(t.get("quoteVolume", 0) or 0)
        if qv <= 0: continue
        scored.append((qv, sym))
    scored.sort(reverse=True)
    return [s for _, s in scored[:TOPK]]

def current_price(ex, symbol: str) -> Optional[float]:
    try:
        o = ex.fetch_ticker(symbol)
        for k in ("last","close","bid","ask"):
            v = o.get(k)
            if v: return float(v)
    except Exception as e:
        log_warn(f"price miss {symbol}: {e}")
    return None

def size_to_qty(ex, symbol: str, usd: float) -> Optional[float]:
    m = ex.market(symbol)
    price = current_price(ex, symbol)
    if price is None or price <= 0: return None
    qty = usd / price
    amt_min = (m.get("limits", {}).get("amount", {}).get("min") or 0) or 0
    step_prec = m.get("precision", {}).get("amount")
    if amt_min and qty < amt_min: return None
    if isinstance(step_prec, int):
        qty = math.floor(qty * (10**step_prec)) / (10**step_prec)
    return max(qty, 0)

def simulate_or_place_buy(ex, symbol: str, usd: float) -> Tuple[bool,str]:
    if DRY_RUN:
        log_warn("🚧 DRY RUN — NO REAL ORDERS SENT 🚧")
        return True, f"SIM_BUY {symbol} ${usd:.2f}"
    try:
        qty = size_to_qty(ex, symbol, usd)
        if qty is None or qty <= 0:
            return False, f"qty too small for {symbol}"
        ex.create_order(symbol, "market", "buy", qty)
        return True, f"BUY {symbol} qty={qty}"
    except Exception as e:
        return False, f"buy err {symbol}: {e}"

def simulate_or_place_sell(ex, symbol: str, qty: float) -> Tuple[bool,str]:
    if DRY_RUN:
        log_warn("🚧 DRY RUN — NO REAL ORDERS SENT 🚧")
        return True, f"SIM_SELL {symbol} qty={qty:.8f}"
    try:
        ex.create_order(symbol, "market", "sell", qty)
        return True, f"SELL {symbol} qty={qty}"
    except Exception as e:
        return False, f"sell err {symbol}: {e}"

def update_trailing(high: float, price: float) -> float:
    return max(high, price)

def run():
    print(("🚧 DRY RUN — NO REAL ORDERS SENT 🚧" if DRY_RUN else "✅ LIVE ORDERS ENABLED").center(80, "="))
    ex = get_exchange()
    positions = load_positions()

    realized_today = 0.0
    sells_count = 0
    buys_count = 0

    # -------- EXIT CHECKS --------
    to_remove = []
    for sym, pos in list(positions.items()):
        price = current_price(ex, sym)
        if price is None:
            log_warn(f"skip exit check (no price) {sym}")
            continue

        entry = float(pos["entry"])
        qty   = float(pos["qty"])
        high  = float(pos.get("high", entry))
        pnl_pct = (price - entry) / entry
        high = update_trailing(high, price)
        positions[sym]["high"] = high

        # TAKE PROFIT
        if pnl_pct >= TAKE_PROFIT_PCT:
            ok, msg = simulate_or_place_sell(ex, sym, qty)
            log_good(f"TP exit {sym} @ {price:.6f} ({pnl_pct*100:.2f}%) → {msg}")
            if ok:
                realized_today += (price - entry) * qty
                sells_count += 1
                to_remove.append(sym)
            continue

        # STOP LOSS
        if pnl_pct <= -abs(STOP_LOSS_PCT):
            ok, msg = simulate_or_place_sell(ex, sym, qty)
            log_error(f"SL exit {sym} @ {price:.6f} ({pnl_pct*100:.2f}%) → {msg}")
            if ok:
                realized_today += (price - entry) * qty
                sells_count += 1
                to_remove.append(sym)
            continue

        # TRAILING STOP
        if high > 0 and (price <= high * (1 - abs(TRAIL_PCT))):
            ok, msg = simulate_or_place_sell(ex, sym, qty)
            log_warn(f"TRAIL exit {sym} high={high:.6f} now={price:.6f} → {msg}")
            if ok:
                realized_today += (price - entry) * qty
                sells_count += 1
                to_remove.append(sym)

    for sym in to_remove:
        positions.pop(sym, None)

    # -------- GUARDS --------
    equity_ref = 1.0
    daily_pl_pct = (realized_today / equity_ref) if equity_ref > 0 else 0.0
    allow_buys = daily_pl_pct >= DAILY_LOSS_CAP_PCT
    if not allow_buys:
        log_error(f"Auto-pause: daily realized P/L {daily_pl_pct:.4f} < cap {DAILY_LOSS_CAP_PCT:.4f}")

    usd_free = fetch_usd_free(ex)
    spendable = max(0.0, usd_free - RESERVE_USD)
    if spendable < USD_PER_TRADE:
        log_warn(f"Cash low: free={usd_free:.2f}, reserve={RESERVE_USD:.2f}, spendable={spendable:.2f}")

    # -------- UNIVERSE --------
    if UNIVERSE_MODE.lower() == "auto":
        universe = universe_auto(ex)
        log_good(f"Auto universe: {', '.join(universe) if universe else '∅'}")
    else:
        universe = WHITELIST
        log_good(f"Whitelist: {', '.join(universe)}")

    # -------- ENTRY LOGIC --------
    if len(positions) > MAX_POSITIONS:
        log_warn(f"Over cap positions={len(positions)} > MAX_POSITIONS={MAX_POSITIONS}. (No auto-trim in this minimalist main.py)")

    buys_made = 0
    if allow_buys and spendable >= USD_PER_TRADE and len(positions) < MAX_POSITIONS:
        for sym in universe:
            if buys_made >= MAX_ENTRIES_PER_RUN: break
            if sym in positions: continue
            price = current_price(ex, sym)
            if price is None: continue
            qty = size_to_qty(ex, sym, USD_PER_TRADE)
            if qty is None or qty <= 0:
                log_warn(f"skip buy (qty too small) {sym}")
                continue
            ok, msg = simulate_or_place_buy(ex, sym, USD_PER_TRADE)
            if ok:
                buys_made += 1
                positions[sym] = {
                    "entry": price,
                    "qty": qty if not DRY_RUN else qty,
                    "ts": now_utc_ts(),
                    "high": price
                }
                log_good(f"BUY OPEN {sym} entry={price:.6f} qty={qty:.8f} → {msg}")

    # -------- MARK TO MARKET --------
    unreal = 0.0
    for sym, pos in positions.items():
        price = current_price(ex, sym)
        if price is None: continue
        entry = float(pos["entry"])
        qty   = float(pos["qty"])
        unreal += (price - entry) * qty

    save_positions(positions)

    summary = (
        f"time={now_iso()} dry_run={DRY_RUN} exchange={EXCHANGE_ID}\n"
        f"positions={len(positions)} buys={buys_made} sells={sells_count}\n"
        f"pnl_realized_today={realized_today:.2f} pnl_unrealized_now={unreal:.2f}\n"
    )
    print("\n" + ("-"*60))
    if realized_today >= 0 or (realized_today == 0 and unreal >= 0):
        log_good("SUMMARY\n" + summary)
    elif realized_today < 0:
        log_error("SUMMARY\n" + summary)
    else:
        log_warn("SUMMARY\n" + summary)

    with open(SUMMARY_F, "w", encoding="utf-8") as f:
        f.write(summary)

    append_kpi({
        "ts": str(now_utc_ts()),
        "iso": now_iso(),
        "pnl_realized": f"{realized_today:.2f}",
        "pnl_unrealized": f"{unreal:.2f}",
        "positions": str(len(positions)),
        "buys": str(buys_made),
        "sells": str(sells_count),
        "dry_run": str(DRY_RUN),
    })

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        log_error(f"FATAL: {e}")
        sys.exit(1)
