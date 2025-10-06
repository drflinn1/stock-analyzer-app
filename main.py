# main.py — Crypto Live with Guard Pack (rotation + sell guards + dust filter)
# - Auto-pick Top-K by 24h % change (USD quote by default)
# - Reserve cash, max positions, daily entry cap
# - Rotation guard to reduce churn (knobs below)
# - Minimal TAKE_PROFIT / STOP_LOSS / TRAILing stop with persistent state
# - NEW: DUST_MIN_USD filter (ignore tiny positions when counting/rotating)
#
# Rotation knobs:
#   ROTATE_WHEN_FULL:         "false" to prevent churny flips when portfolio is full (default "false")
#   MAX_BUYS_PER_RUN:         limit new entries/rotations per run (default "1")
#   ROTATE_WHEN_CASH_SHORT:   if "true", allow targeted rotation when cash is short (default "true")
#
# Sell guards (tokens for CI): TAKE_PROFIT / STOP_LOSS / TRAIL (trailing)
#   TP_PCT, SL_PCT, TSL_PCT
#
# Dust:
#   DUST_MIN_USD: minimum per-position USD value to *count/manage* (default "2")
#   Positions below this are ignored in counts and rotation decisions.

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

QUOTE            = os.getenv("QUOTE", "USD")
WHITELIST        = [s.strip() for s in os.getenv("SYMBOLS_WHITELIST","").split(",") if s.strip()]
EXCLUDE          = [s.strip() for s in os.getenv("SYMBOLS_EXCLUDE","").split(",") if s.strip()]

TOP_K            = as_int(os.getenv("TOP_K"), 6)
MAX_POSITIONS    = as_int(os.getenv("MAX_POSITIONS"), 6)
MIN_NOTIONAL     = as_float(os.getenv("MIN_NOTIONAL"), 5.0)

RESERVE_CASH_PCT = as_float(os.getenv("RESERVE_CASH_PCT"), 0.05)
DRY_RUN          = as_bool(os.getenv("DRY_RUN"), True)
RUN_SWITCH       = as_bool(os.getenv("RUN_SWITCH"), True)

# SELL GUARDS (tokens for CI present below)
TP_PCT           = as_float(os.getenv("TP_PCT"), 0.0)    # TAKE_PROFIT percent
SL_PCT           = as_float(os.getenv("SL_PCT"), 0.0)    # STOP_LOSS percent
TSL_PCT          = as_float(os.getenv("TSL_PCT"), 0.0)   # TRAIL / trailing percent

DAILY_LOSS_CAP_PCT  = as_float(os.getenv("DAILY_LOSS_CAP_PCT"), 0.0)
MAX_DAILY_ENTRIES   = as_int(os.getenv("MAX_DAILY_ENTRIES"), 9999)

# Rotation knobs
ROTATE_WHEN_FULL       = as_bool(os.getenv("ROTATE_WHEN_FULL"), False)
MAX_BUYS_PER_RUN       = as_int(os.getenv("MAX_BUYS_PER_RUN"), 1)
ROTATE_WHEN_CASH_SHORT = as_bool(os.getenv("ROTATE_WHEN_CASH_SHORT"), True)

# NEW: dust filter
DUST_MIN_USD        = as_float(os.getenv("DUST_MIN_USD"), 2.0)

STATE_DIR        = os.getenv("STATE_DIR", ".state")
os.makedirs(STATE_DIR, exist_ok=True)

# ---------- CCXT Setup ---------- #
try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

def connect_exchange():
    cls = getattr(ccxt, EXCHANGE_ID)
    opts = { "apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True }
    if PASSWORD:
        opts["password"] = PASSWORD
    return cls(opts)

# ---------- IO Utils ---------- #

def read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default

def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# ---------- Market / Price ---------- #

def load_markets_safe(ex):
    if not getattr(ex, "markets", None):
        ex.load_markets()
    return ex.markets

def fetch_last_price(ex, symbol: str) -> float:
    t = ex.fetch_ticker(symbol)
    if t and t.get("last"):
        return float(t["last"])
    bid = float(t.get("bid") or 0)
    ask = float(t.get("ask") or 0)
    if bid and ask:
        return (bid+ask)/2
    raise RuntimeError(f"No price for {symbol}")

def fetch_tickers_24h_change(ex, quote: str) -> List[Tuple[str,float]]:
    tickers = {}
    try:
        tickers = ex.fetch_tickers()
    except Exception:
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
            try:
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

def filter_dust_positions(ex, positions: Dict[str,float], dust_min_usd: float) -> Dict[str,float]:
    """Keep only positions whose USD value >= dust_min_usd."""
    if dust_min_usd <= 0:
        return positions
    filtered: Dict[str,float] = {}
    for sym, amt in positions.items():
        try:
            price = fetch_last_price(ex, sym)
            usd_val = amt * price
            if usd_val >= dust_min_usd:
                filtered[sym] = amt
            else:
                yellow(f"Ignoring dust {sym}: ${usd_val:.2f} < DUST_MIN_USD={dust_min_usd:.2f}")
        except Exception:
            # if price fetch fails, keep it (safer) — or skip; choose skip to avoid count inflation
            yellow(f"Price missing for {sym}; treating as dust and skipping.")
    return filtered

# ---------- Daily Entry Cap ---------- #

def daily_entry_counter_path() -> str:
    return os.path.join(STATE_DIR, "daily_entries.json")

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
    return ex.create_order(symbol, type="market", side="buy", amount=qty)

def place_market_sell(ex, symbol: str, amount: float) -> Optional[Dict[str,Any]]:
    if amount <= 0:
        return None
    last = fetch_last_price(ex, symbol)
    if DRY_RUN:
        yellow(f"🚧 DRY RUN — SELL {symbol} amount {amount:.6f} at {last:.6f}")
        return {"id":"SIM-SELL","symbol":symbol,"amount":amount,"price":last}
    return ex.create_order(symbol, type="market", side="sell", amount=amount)

# ---------- Sell Guard State ---------- #

POS_STATE_PATH   = os.path.join(STATE_DIR, "positions_state.json")
KPI_PATH         = os.path.join(STATE_DIR, "kpi_history.csv")

def load_pos_state() -> Dict[str, Dict[str,float]]:
    return read_json(POS_STATE_PATH, {})

def save_pos_state(state: Dict[str, Dict[str,float]]) -> None:
    write_json(POS_STATE_PATH, state)

def ensure_state_for_holds(state: Dict[str,Dict[str,float]], ex, holds: Dict[str,float]) -> None:
    changed = False
    for sym in holds.keys():
        if sym not in state:
            price = fetch_last_price(ex, sym)
            state[sym] = {"entry": float(price), "peak": float(price)}
            changed = True
    if changed:
        save_pos_state(state)

# ---------- Sell Rules (CI token lines present) ---------- #

def apply_take_profit(symbol: str, price: float, entry: float) -> bool:
    """TAKE_PROFIT check — returns True if we should sell."""
    if TP_PCT <= 0:
        return False
    target = entry * (1.0 + TP_PCT/100.0)
    return price >= target

def apply_stop_loss(symbol: str, price: float, entry: float) -> bool:
    """STOP_LOSS check — returns True if we should sell."""
    if SL_PCT <= 0:
        return False
    floor = entry * (1.0 - SL_PCT/100.0)
    return price <= floor

def apply_trailing(symbol: str, price: float, peak: float) -> Tuple[bool, float]:
    """TRAIL / trailing stop — returns (should_sell, new_peak)."""
    if TSL_PCT <= 0:
        return (False, max(peak, price))
    new_peak = max(peak, price)
    if new_peak > 0 and (new_peak - price) / new_peak * 100.0 >= TSL_PCT:
        return (True, new_peak)
    return (False, new_peak)

# ---------- Strategy Core ---------- #

def choose_candidates(ex, quote: str) -> List[str]:
    ranked = fetch_tickers_24h_change(ex, quote)
    take = ranked[:TOP_K]
    if not take:
        yellow("No candidates from 24h change ranking.")
    return [s for s,_ in take]

def pick_worst_symbol(ex, positions: Dict[str,float]) -> Optional[str]:
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

    # Portfolio snapshot
    cash, positions = get_cash_and_positions(ex, QUOTE)
    # Apply dust filter before counting/rotation decisions
    positions = filter_dust_positions(ex, positions, DUST_MIN_USD)
    pos_count = len(positions)
    green(f"Cash {QUOTE}: {cash:.2f} | Positions: {pos_count}/{MAX_POSITIONS}")

    # Ensure sell-guard state for anything we already hold
    pos_state = load_pos_state()
    ensure_state_for_holds(pos_state, ex, positions)

    # ---------- SELL PASS (TAKE_PROFIT / STOP_LOSS / TRAIL) ----------
    sells_this_run = 0
    for sym, amt in list(positions.items()):
        price = fetch_last_price(ex, sym)
        st = pos_state.get(sym, {"entry": price, "peak": price})
        entry = float(st.get("entry", price))
        peak  = float(st.get("peak", entry))

        should_trail_sell, new_peak = apply_trailing(sym, price, peak)
        st["peak"] = float(new_peak)
        pos_state[sym] = st

        tp = apply_take_profit(sym, price, entry)
        sl = apply_stop_loss(sym, price, entry)
        tr = should_trail_sell

        if tp or sl or tr:
            reason = "TAKE_PROFIT" if tp else ("STOP_LOSS" if sl else "TRAIL")
            yellow(f"{reason}: Selling {sym} @ {price:.6f} (entry {entry:.6f}, peak {new_peak:.6f})")
            place_market_sell(ex, sym, amt)
            sells_this_run += 1
            positions.pop(sym, None)
            pos_state.pop(sym, None)

    save_pos_state(pos_state)

    # ---------- BUY / ROTATION PASS ----------
    candidates = choose_candidates(ex, QUOTE)

    def usable_cash():
        return max(0.0, cash * (1.0 - RESERVE_CASH_PCT))

    buys_this_run = 0

    for sym in candidates:
        if buys_this_run >= MAX_BUYS_PER_RUN:
            yellow(f"Reached MAX_BUYS_PER_RUN={MAX_BUYS_PER_RUN}; skipping further buys.")
            break

        at_capacity = (pos_count >= MAX_POSITIONS)
        need_cash = (usable_cash() < MIN_NOTIONAL)

        if at_capacity:
            if not ROTATE_WHEN_FULL:
                if not (need_cash and ROTATE_WHEN_CASH_SHORT):
                    yellow(f"Portfolio full and ROTATE_WHEN_FULL=false; skipping rotation for {sym}.")
                    continue
            worst = pick_worst_symbol(ex, positions)
            if worst and worst != sym:
                amt = positions.get(worst, 0.0)
                if amt > 0:
                    yellow(f"Rotating: selling worst {worst} to enter {sym}")
                    place_market_sell(ex, worst, amt)
                    positions.pop(worst, None)
                    pos_count -= 1
                    cash, positions2 = get_cash_and_positions(ex, QUOTE)
                    positions2 = filter_dust_positions(ex, positions2, DUST_MIN_USD)
                    # refresh local snapshot
                    positions.update(positions2)
                    # cash updated by re-fetch above

        if usable_cash() < MIN_NOTIONAL:
            yellow(f"Not enough usable cash for {sym}; skipping.")
            continue

        if sym in positions:
            continue

        if not under_daily_entry_cap():
            yellow("Hit MAX_DAILY_ENTRIES cap for today; no more buys.")
            break

        slots_left = max(1, MAX_POSITIONS - pos_count)
        budget = max(MIN_NOTIONAL, usable_cash() / float(slots_left))

        resp = place_market_buy(ex, sym, budget)
        if resp:
            buys_this_run += 1
            pos_count += 1
            bump_daily_entry()
            price = fetch_last_price(ex, sym)
            pos_state = load_pos_state()
            pos_state[sym] = {"entry": float(price), "peak": float(price)}
            save_pos_state(pos_state)

    # ---------- KPI / Summary ----------
    ranked = fetch_tickers_24h_change(ex, QUOTE)
    held_changes = {s:c for s,c in ranked if s in positions}
    avg_chg = (sum(held_changes.values())/max(1,len(held_changes))) if held_changes else 0.0

    print("-"*78)
    green(f"SUMMARY: positions={pos_count}  buys_this_run={buys_this_run}  sells_this_run={sells_this_run}  avg_24h_change={avg_chg:.2f}%")
    if DRY_RUN:
        yellow("🚧 DRY RUN — NO REAL ORDERS SENT 🚧")
    print("-"*78)

    # KPI CSV
    try:
        write_header = not os.path.exists(KPI_PATH)
        with open(KPI_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["ts_utc","pos_count","buys_this_run","sells_this_run","avg_24h_change","dry_run"])
            w.writerow([now_utc(), pos_count, buys_this_run, sells_this_run, f"{avg_chg:.4f}", int(DRY_RUN)])
    except Exception as e:
        yellow(f"KPI CSV write failed: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        red(f"Unhandled error: {e}")
        raise
