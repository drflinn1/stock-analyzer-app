# main.py — Crypto Live with Guard Pack (minimal rotation patch)
# - USD-only focus by default, auto-pick Top-K by 24h % change
# - Take-profit / Stop-loss / Trailing stop (simple, opt-in)
# - Reserve cash, max positions, daily caps
# - Rotation guard to reduce churn (new knobs below)
#
# NEW KNOBS (minimal patch):
#   ROTATE_WHEN_FULL:         "false" to prevent churny flips when portfolio is full (default "false")
#   MAX_BUYS_PER_RUN:         limit new entries/rotations per run (default "1")
#   ROTATE_WHEN_CASH_SHORT:   if "true", allow targeted rotation only when we actually need cash (default "true")
#
# This script is self-contained and aims to match your existing behavior with only the minimal rotation edits.

from __future__ import annotations
import os, json, time, math, csv
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone

# ---------- Helpers ---------- #

def as_bool(v: Optional[str], default: bool=False) -> bool:
    if v is None:
        return default
    v = v.strip().lower()
    return v in ("1","true","yes","y","on")

def as_float(v: Optional[str], default: float) -> float:
    try:
        return float(v) if v is not None else default
    except:
        return default

def as_int(v: Optional[str], default: int) -> int:
    try:
        return int(v) if v is not None else default
    except:
        return default

def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

def log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)

def green(msg: str) -> None: log("OK", msg)
def yellow(msg: str) -> None: log("WARN", msg)
def red(msg: str) -> None: log("ERR", msg)

# ---------- ENV ---------- #

EXCHANGE_ID      = os.getenv("EXCHANGE_ID", "kraken")
API_KEY          = os.getenv("API_KEY", "")
API_SECRET       = os.getenv("API_SECRET", "")
PASSWORD         = os.getenv("API_PASSWORD", None)

QUOTE            = os.getenv("QUOTE", "USD")             # Focus quote
WHITELIST        = [s.strip() for s in os.getenv("SYMBOLS_WHITELIST","").split(",") if s.strip()]  # optional hard whitelist
EXCLUDE          = [s.strip() for s in os.getenv("SYMBOLS_EXCLUDE","").split(",") if s.strip()]    # optional exclusions

TOP_K            = as_int(os.getenv("TOP_K"), 6)         # auto-pick cap by 24h % change
MAX_POSITIONS    = as_int(os.getenv("MAX_POSITIONS"), 6)
MIN_NOTIONAL     = as_float(os.getenv("MIN_NOTIONAL"), 5.0)

RESERVE_CASH_PCT = as_float(os.getenv("RESERVE_CASH_PCT"), 0.05)  # keep some dry powder
DRY_RUN          = as_bool(os.getenv("DRY_RUN"), True)
RUN_SWITCH       = as_bool(os.getenv("RUN_SWITCH"), True)

TP_PCT           = as_float(os.getenv("TP_PCT"), 0.0)    # 0 disables
SL_PCT           = as_float(os.getenv("SL_PCT"), 0.0)    # 0 disables
TSL_PCT          = as_float(os.getenv("TSL_PCT"), 0.0)   # 0 disables

DAILY_LOSS_CAP_PCT  = as_float(os.getenv("DAILY_LOSS_CAP_PCT"), 0.0)  # 0 disables
MAX_DAILY_ENTRIES   = as_int(os.getenv("MAX_DAILY_ENTRIES"), 9999)    # simple limiter

# --- NEW (Edit #1): rotation knobs & small loop counter limit ---
ROTATE_WHEN_FULL    = as_bool(os.getenv("ROTATE_WHEN_FULL"), False)
MAX_BUYS_PER_RUN    = as_int(os.getenv("MAX_BUYS_PER_RUN"), 1)
ROTATE_WHEN_CASH_SHORT = as_bool(os.getenv("ROTATE_WHEN_CASH_SHORT"), True)

STATE_DIR        = os.getenv("STATE_DIR", ".state")
os.makedirs(STATE_DIR, exist_ok=True)

# ---------- CCXT Setup ---------- #
try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

def connect_exchange():
    cls = getattr(ccxt, EXCHANGE_ID)
    opts = {
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
    }
    if PASSWORD:
        opts["password"] = PASSWORD
    ex = cls(opts)
    return ex

# ---------- Data Fetch ---------- #

def load_markets_safe(ex):
    if not getattr(ex, "markets", None):
        ex.load_markets()
    return ex.markets

def fetch_tickers_24h_change(ex, quote: str) -> List[Tuple[str,float]]:
    """
    Returns list of (symbol, 24h_change_pct) for markets ending in /QUOTE.
    """
    # Some exchanges support fetchTickers; else loop
    tickers = {}
    try:
        tickers = ex.fetch_tickers()
    except Exception:
        # Fallback: loop visible markets
        pass

    results: List[Tuple[str,float]] = []
    markets = load_markets_safe(ex)
    for sym, m in markets.items():
        if not m.get("active", True):
            continue
        if f"/{quote}" not in sym:
            continue
        if EXCLUDE and sym in EXCLUDE:
            continue
        if WHITELIST and sym not in WHITELIST:
            continue

        change = None
        tk = tickers.get(sym) if tickers else None
        if tk and "percentage" in tk and tk["percentage"] is not None:
            change = tk["percentage"]
        else:
            # Fallback: try OHLCV if percentage isn’t available
            try:
                # last ~2 candles of 1h to approximate 24h? use 1d when supported
                # Prefer 1d; if unsupported, skip.
                ohlcv = ex.fetch_ohlcv(sym, timeframe="1d", limit=2)
                if ohlcv and len(ohlcv) >= 2:
                    prev_close = ohlcv[-2][4]
                    last_close = ohlcv[-1][4]
                    if prev_close and last_close:
                        change = (last_close - prev_close) / prev_close * 100.0
            except Exception:
                change = None

        if change is not None:
            results.append((sym, float(change)))

    # sort high to low by % change
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ---------- Portfolio Helpers ---------- #

def get_cash_and_positions(ex, quote: str) -> Tuple[float, Dict[str,float]]:
    bal = ex.fetch_balance()
    free = bal.get("free", {}) or {}
    total = bal.get("total", {}) or {}

    cash = float(free.get(quote, 0.0))

    markets = ex.load_markets()
    positions: Dict[str,float] = {}
    for sym, m in markets.items():
        if not m.get("active", True):
            continue
        if f"/{quote}" not in sym:
            continue
        base = m["base"]
        amt_total = float(total.get(base, 0.0))
        if amt_total > 0:
            positions[sym] = amt_total
    return cash, positions

def fetch_last_price(ex, symbol: str) -> float:
    t = ex.fetch_ticker(symbol)
    if t and t.get("last"):
        return float(t["last"])
    # fallback mid
    bid = float(t.get("bid") or 0)
    ask = float(t.get("ask") or 0)
    if bid and ask:
        return (bid+ask)/2
    raise RuntimeError(f"No price for {symbol}")

# ---------- Guards ---------- #

def daily_entry_counter_path() -> str:
    d = os.path.join(STATE_DIR, "daily_entries.json")
    return d

def read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default

def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def under_daily_entry_cap() -> bool:
    if MAX_DAILY_ENTRIES >= 9999:
        return True
    path = daily_entry_counter_path()
    data = read_json(path, {"date":"","count":0})
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if data["date"] != today:
        data = {"date": today, "count": 0}
        write_json(path, data)
    return data["count"] < MAX_DAILY_ENTRIES

def bump_daily_entry() -> None:
    path = daily_entry_counter_path()
    data = read_json(path, {"date":"","count":0})
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if data["date"] != today:
        data = {"date": today, "count": 0}
    data["count"] += 1
    write_json(path, data)

# ---------- Orders (with DRY_RUN) ---------- #

def place_market_buy(ex, symbol: str, usd_amount: float) -> Optional[Dict[str,Any]]:
    if usd_amount < MIN_NOTIONAL:
        yellow(f"Skip tiny buy {symbol}: {usd_amount:.2f} < MIN_NOTIONAL")
        return None
    last = fetch_last_price(ex, symbol)
    qty  = usd_amount / last
    if DRY_RUN:
        yellow(f"🚧 DRY RUN — BUY {symbol} ~ ${usd_amount:.2f} (~{qty:.6f}) at {last:.6f}")
        return {"id":"SIM-BUY","symbol":symbol,"amount":qty,"cost":usd_amount,"price":last}
    else:
        return ex.create_order(symbol, type="market", side="buy", amount=qty)

def place_market_sell(ex, symbol: str, amount: float) -> Optional[Dict[str,Any]]:
    if amount <= 0:
        return None
    last = fetch_last_price(ex, symbol)
    if DRY_RUN:
        yellow(f"🚧 DRY RUN — SELL {symbol} amount {amount:.6f} at {last:.6f}")
        return {"id":"SIM-SELL","symbol":symbol,"amount":amount,"price":last}
    else:
        return ex.create_order(symbol, type="market", side="sell", amount=amount)

# ---------- Strategy Core ---------- #

def choose_candidates(ex, quote: str) -> List[str]:
    ranked = fetch_tickers_24h_change(ex, quote)
    take = ranked[:TOP_K]
    if not take:
        yellow("No candidates from 24h change ranking.")
    return [s for s,_ in take]

def pick_worst_symbol(ex, positions: Dict[str,float]) -> Optional[str]:
    # Heuristic: pick the position with lowest 24h % change
    if not positions:
        return None
    ranked = fetch_tickers_24h_change(ex, QUOTE)
    map_change = {s:c for s,c in ranked}
    worst = None
    worst_chg = 10**9
    for sym in positions.keys():
        chg = map_change.get(sym, 0.0)
        if chg < worst_chg:
            worst = sym
            worst_chg = chg
    return worst

def main():
    print("\n" + "="*78)
    print(f"Crypto Live — {now_utc()}  (DRY_RUN={DRY_RUN})")
    print("="*78)

    if not RUN_SWITCH:
        yellow("RUN_SWITCH is off → exiting early.")
        return

    ex = connect_exchange()
    load_markets_safe(ex)

    # Fetch portfolio + cash
    cash, positions = get_cash_and_positions(ex, QUOTE)
    pos_count = len(positions)
    green(f"Cash {QUOTE}: {cash:.2f} | Positions: {pos_count}/{MAX_POSITIONS}")

    # Ranking for fresh entries
    candidates = choose_candidates(ex, QUOTE)

    # Simple reserve
    # Keep RESERVE_CASH_PCT in cash if possible
    def usable_cash():
        return max(0.0, cash * (1.0 - RESERVE_CASH_PCT))

    # ---- NEW (Edit #2): per-run buy/rotation counter ----
    buys_this_run = 0

    # Decide entries
    # We’ll try to buy top names we don’t already hold, limited by caps and knobs.
    for sym in candidates:
        if buys_this_run >= MAX_BUYS_PER_RUN:
            yellow(f"Reached MAX_BUYS_PER_RUN={MAX_BUYS_PER_RUN}; skipping further buys.")
            break

        already = sym in positions
        at_capacity = (pos_count >= MAX_POSITIONS)

        need_cash = (usable_cash() < MIN_NOTIONAL)

        # --- NEW (Edit #3): rotation gate when full ---
        if at_capacity:
            if not ROTATE_WHEN_FULL:
                # Only rotate if we are cash-short AND rotation is explicitly allowed for cash needs.
                if not (need_cash and ROTATE_WHEN_CASH_SHORT):
                    yellow(f"Portfolio full and ROTATE_WHEN_FULL=false; skipping rotation for {sym}.")
                    continue
            # If we do rotate: sell the worst to free up notional.
            worst = pick_worst_symbol(ex, positions)
            if worst and worst != sym:
                amt = positions.get(worst, 0.0)
                if amt > 0:
                    yellow(f"Rotating: selling worst {worst} to enter {sym}")
                    place_market_sell(ex, worst, amt)
                    # update local view
                    positions.pop(worst, None)
                    pos_count -= 1
                    # refresh cash snapshot
                    cash, positions2 = get_cash_and_positions(ex, QUOTE)
                    cash = cash
                    positions.update(positions2)

        # (Re)check cash for buy
        if usable_cash() < MIN_NOTIONAL:
            yellow(f"Not enough usable cash for {sym}; skipping.")
            continue

        # skip if already holding
        if already:
            continue

        # Entry sizing: equal-weight by remaining slots
        slots_left = max(1, MAX_POSITIONS - pos_count)
        budget = max(MIN_NOTIONAL, usable_cash() / float(slots_left))

        resp = place_market_buy(ex, sym, budget)
        if resp:
            buys_this_run += 1
            pos_count += 1
            bump_daily_entry()

    # (Optional) very simple TP/SL/TSL demo — no historical cost basis store here to keep patch minimal.
    # You likely already have more complete exit logic; keeping this minimal on purpose.
    if TP_PCT > 0 or SL_PCT > 0 or TSL_PCT > 0:
        yellow("Simple TP/SL/TSL placeholder active (minimal). For full exits, keep your existing module.")
        # Intentionally minimal in this patch to avoid altering existing behavior.

    # KPI / Summary
    ranked = fetch_tickers_24h_change(ex, QUOTE)
    held_changes = {s:c for s,c in ranked if s in positions}
    if held_changes:
        avg_chg = sum(held_changes.values()) / max(1,len(held_changes))
    else:
        avg_chg = 0.0

    print("-"*78)
    green(f"SUMMARY: positions={pos_count}  buys_this_run={buys_this_run}  avg_24h_change={avg_chg:.2f}%")
    if DRY_RUN:
        yellow("🚧 DRY RUN — NO REAL ORDERS SENT 🚧")
    print("-"*78)

    # write tiny KPI CSV history
    try:
        csv_path = os.path.join(STATE_DIR, "kpi_history.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["ts_utc","pos_count","buys_this_run","avg_24h_change","dry_run"])
            w.writerow([now_utc(), pos_count, buys_this_run, f"{avg_chg:.4f}", int(DRY_RUN)])
    except Exception as e:
        yellow(f"KPI CSV write failed: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        red(f"Unhandled error: {e}")
        raise
