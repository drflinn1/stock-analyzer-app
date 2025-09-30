# trader/main.py — Crypto Live with Guard Pack
# - USD-only whitelist (BTC/ETH/SOL/DOGE)
# - Auto-pick Top-K with cooldown
# - Take-profit / Stop-loss / Trailing stop
# - Reserve cash, daily loss cap, max daily entries
# - Emergency SL, auto-pause guard, KPI/CSV, DRY-RUN banner
#
# This file is designed to be drop-in and self-contained. It will run with sane
# defaults even if your workflow inputs/envs are not set. All configuration can
# be controlled via environment variables (see ENV section below).

from __future__ import annotations
import os, json, time, random, math, csv
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

# ---------- ENV & DEFAULTS ---------- #

def as_bool(val: Optional[str], default: bool) -> bool:
    if val is None: return default
    return str(val).strip().lower() in {"1","true","yes","on"}

def as_float(val: Optional[str], default: float) -> float:
    try:
        return float(val) if val is not None and str(val).strip() != "" else default
    except: return default

def as_int(val: Optional[str], default: int) -> int:
    try:
        return int(float(val)) if val is not None and str(val).strip() != "" else default
    except: return default

DRY_RUN = as_bool(os.getenv("DRY_RUN"), True)
RUN_SWITCH = os.getenv("RUN_SWITCH", "ON").upper()  # "ON" or "OFF"
USD_PER_TRADE = as_float(os.getenv("USD_PER_TRADE"), 10.0)
MAX_POSITIONS = as_int(os.getenv("MAX_POSITIONS"), 3)

TP_PCT = as_float(os.getenv("TP_PCT"), 3.5)        # %
SL_PCT = as_float(os.getenv("SL_PCT"), 2.0)        # %
TRAIL_PCT = as_float(os.getenv("TRAIL_PCT"), 1.2)  # %

# Guard Pack parameters
RESERVE_USD = as_float(os.getenv("RESERVE_USD"), 100.0)
DAILY_LOSS_CAP_USD = as_float(os.getenv("DAILY_LOSS_CAP_USD"), 5.0)
DAILY_LOSS_CAP_PCT = as_float(os.getenv("DAILY_LOSS_CAP_PCT"), 2.0)  # percent of start cash
MAX_ENTRIES_PER_DAY = as_int(os.getenv("MAX_ENTRIES_PER_DAY"), 6)
COOLDOWN_MIN = as_int(os.getenv("COOLDOWN_MIN"), 120)  # ~ 8 runs if 15m cadence
TOPK = max(1, as_int(os.getenv("TOPK"), 3))
PICK_MODE = os.getenv("PICK_MODE", "random").lower()  # "random" or "rotate"
EMERGENCY_SL_PCT = as_float(os.getenv("EMERGENCY_SL_PCT"), 3.5)

STATE_DIR = ".state"
POSITIONS_FILE = os.path.join(STATE_DIR, "positions.json")
DAY_FILE = os.path.join(STATE_DIR, "day_state.json")
COOLDOWN_FILE = os.path.join(STATE_DIR, "cooldown.json")
PAUSE_FILE = os.path.join(STATE_DIR, "guard_pause.json")
HISTORY_CSV = os.path.join(STATE_DIR, "kpi_history.csv")
ROTATE_FILE = os.path.join(STATE_DIR, "rotate_idx.json")

USD_KEYS = ("USD", "ZUSD")
UNIVERSE = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]

os.makedirs(STATE_DIR, exist_ok=True)

# ---------- UTIL ---------- #

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default

def save_json(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def append_history(row: Dict[str, Any]):
    header = ["ts","dry_run","open","cap_left","usd","pnl_today","entries_today","guards"]
    newfile = not os.path.exists(HISTORY_CSV)
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if newfile:
            w.writeheader()
        w.writerow({k: row.get(k) for k in header})

# ---------- EXCHANGE ---------- #

def get_exchange() -> Any:
    api_key = os.getenv("KRAKEN_API_KEY")
    api_secret = os.getenv("KRAKEN_API_SECRET")
    cfg = {"enableRateLimit": True}
    if not DRY_RUN and api_key and api_secret:
        cfg.update({"apiKey": api_key, "secret": api_secret})
    return ccxt.kraken(cfg)

# ---------- BALANCES ---------- #

def get_usd_balance(ex) -> float:
    bal = ex.fetch_balance({})
    total = 0.0
    for k in USD_KEYS:
        if k in bal.get("total", {}):
            total += float(bal["total"][k] or 0)
        elif k in bal:
            total += float(bal.get(k) or 0)
    return float(total)

# ---------- PRICES / RANKING ---------- #

def last_price(ex, symbol: str) -> Optional[float]:
    try:
        t = ex.fetch_ticker(symbol)
        px = t.get("last") or t.get("close") or t.get("bid") or t.get("ask")
        return float(px) if px else None
    except Exception:
        return None

def momentum_score(ex, symbol: str, tf: str = "15m", lookback: int = 12) -> float:
    """Simple 15m momentum over ~3 hours. Fallback to 0 on failure."""
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, limit=lookback)
        if not ohlcv or len(ohlcv) < 2:
            return 0.0
        first = float(ohlcv[0][4])
        last = float(ohlcv[-1][4])
        return (last - first) / first
    except Exception:
        return 0.0

# ---------- SELECTION HELPERS ---------- #

def load_positions() -> List[Dict[str, Any]]:
    return load_json(POSITIONS_FILE, [])

def save_positions(positions: List[Dict[str, Any]]):
    save_json(POSITIONS_FILE, positions)

def load_cooldown() -> Dict[str, float]:
    data = load_json(COOLDOWN_FILE, {})
    now = utcnow().timestamp()
    # prune expired
    data = {sym: exp for sym, exp in data.items() if exp > now}
    save_json(COOLDOWN_FILE, data)
    return data

def set_cooldown(sym: str):
    data = load_json(COOLDOWN_FILE, {})
    exp = (utcnow() + timedelta(minutes=COOLDOWN_MIN)).timestamp()
    data[sym] = exp
    save_json(COOLDOWN_FILE, data)

# ---------- DAY STATE ---------- #

def load_day_state(now_usd: float) -> Dict[str, Any]:
    today = utcnow().date().isoformat()
    ds = load_json(DAY_FILE, {})
    if ds.get("date") != today:
        ds = {
            "date": today,
            "start_cash": now_usd,
            "pnl_today": 0.0,
            "entries_today": 0,
        }
        # clear auto-pause on new day
        if os.path.exists(PAUSE_FILE):
            try: os.remove(PAUSE_FILE)
            except: pass
    save_json(DAY_FILE, ds)
    return ds

# ---------- BROKER OPS (SIM / LIVE) ---------- #

def market_buy(ex, symbol: str, usd: float) -> Tuple[float, float]:
    px = last_price(ex, symbol) or 0.0
    amt = 0.0 if px <= 0 else usd / px
    if DRY_RUN:
        print(f"{ts()} INFO: [DRY RUN] ccxt BUY {symbol} amount={amt} (~${usd:.2f}) @~{px}")
        return px, amt
    # LIVE
    try:
        order = ex.create_market_buy_order(symbol, amt)
        # derive avg price/amount from order
        px2 = float(order.get("average") or px)
        amt2 = float(order.get("amount") or amt)
        print(f"{ts()} INFO: LIVE BUY executed: {symbol} ${usd:.2f} avg~{px2}")
        return px2, amt2
    except Exception as e:
        print(f"{ts()} ERROR: BUY failed: {e}")
        return px, amt

def market_sell(ex, symbol: str, qty: float) -> float:
    px = last_price(ex, symbol) or 0.0
    if DRY_RUN:
        print(f"{ts()} INFO: [DRY RUN] ccxt SELL {symbol} qty={qty}")
        print(f"{ts()} INFO: SELL executed: {symbol} qty={qty}")
        return px
    try:
        order = ex.create_market_sell_order(symbol, qty)
        px2 = float(order.get("average") or px)
        print(f"{ts()} INFO: LIVE SELL executed: {symbol} qty={qty} avg~{px2}")
        return px2
    except Exception as e:
        print(f"{ts()} ERROR: SELL failed: {e}")
        return px

# ---------- TRADING LOGIC ---------- #

def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

class Position:
    def __init__(self, symbol: str, qty: float, entry: float, tp: float, sl: float, trail: float):
        self.symbol = symbol
        self.qty = qty
        self.entry = entry
        self.high = entry
        self.tp_pct = tp
        self.sl_pct = sl
        self.trail_pct = trail
        self.ts = utcnow().isoformat()

    @staticmethod
    def from_d(d: Dict[str, Any]) -> 'Position':
        p = Position(d["symbol"], float(d["qty"]), float(d["entry"]), float(d["tp_pct"]), float(d["sl_pct"]), float(d["trail_pct"]))
        p.high = float(d.get("high", p.entry))
        p.ts = d.get("ts", p.ts)
        return p

    def to_d(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "entry": self.entry,
            "high": self.high,
            "tp_pct": self.tp_pct,
            "sl_pct": self.sl_pct,
            "trail_pct": self.trail_pct,
            "ts": self.ts,
        }

    def check_exits(self, price: float) -> Optional[str]:
        # update trailing high
        if price > self.high:
            self.high = price
        # compute thresholds
        change = (price - self.entry) / self.entry * 100.0
        trail_stop = self.high * (1.0 - self.trail_pct/100.0)
        # emergency first
        if change <= -EMERGENCY_SL_PCT:
            return f"EMERGENCY_SL {EMERGENCY_SL_PCT:.2f}%"
        if change >= self.tp_pct:
            return f"TAKE_PROFIT hit at {self.tp_pct:.2f}%"
        if change <= -self.sl_pct:
            return f"STOP_LOSS hit at {self.sl_pct:.2f}%"
        if price <= trail_stop and self.high > self.entry:
            fall = (price - self.high)/self.high * 100.0
            return f"TRAIL_STOP hit at {abs(fall):.2f}% from high"
        return None

# ---------- MAIN RUN ---------- #

def main():
    print("=========================================================")
    print("🚧 DRY RUN — NO REAL ORDERS SENT 🚧" if DRY_RUN else "⚠️ LIVE MODE — REAL ORDERS ENABLED ⚠️")
    print("=========================================================")

    if RUN_SWITCH.upper() == "OFF":
        print(f"{ts()} WARN: RUN_SWITCH=OFF — exiting early.")
        return

    ex = get_exchange()
    print(f"{ts()} INFO: Starting trader in CRYPTO mode. Dry run={DRY_RUN}. Broker=ccxt")
    print(f"{ts()} INFO: Using crypto broker class: CryptoBroker")

    # Balances
    try:
        usd = get_usd_balance(ex)
    except Exception as e:
        print(f"{ts()} ERROR: Balance fetch failed: {e}")
        usd = 0.0
    print(f"{ts()} INFO: [ccxt] USD balance detected: ${usd:.2f}")
    print(f"{ts()} INFO: Balance via get_balance: ${usd:.2f}")
    print(f"{ts()} INFO: Available balance: ${usd:.2f}")

    # Load state
    day = load_day_state(usd)
    positions = [Position.from_d(p) for p in load_positions()]
    cooldown = load_cooldown()

    # Universe ranking
    universe_scores = []
    for sym in UNIVERSE:
        sc = momentum_score(ex, sym, tf="15m", lookback=12)
        universe_scores.append((sym, sc))
    universe_scores.sort(key=lambda x: x[1], reverse=True)
    ranked = [s for s,_ in universe_scores]
    print(f"{ts()} INFO: Universe: {ranked}")

    # Evaluate exits for open positions
    realized_pnl = 0.0
    still_open: List[Position] = []
    for pos in positions:
        px = last_price(ex, pos.symbol) or pos.entry
        reason = pos.check_exits(px)
        if reason:
            print(f"{ts()} INFO: {reason}")
            sell_px = market_sell(ex, pos.symbol, pos.qty)
            pnl = (sell_px - pos.entry) * pos.qty
            realized_pnl += pnl
            set_cooldown(pos.symbol)
        else:
            still_open.append(pos)
    positions = still_open

    # Update day PnL & entries
    day["pnl_today"] = round(float(day.get("pnl_today", 0.0)) + realized_pnl, 2)
    save_json(DAY_FILE, day)

    # GUARDS
    guards = {
        "reserve": False,
        "daily_cap": False,
        "cooldown": False,
        "max_entries": False,
        "paused": False,
    }

    # Auto-pause file present?
    pause = load_json(PAUSE_FILE, {})
    if pause.get("date") == utcnow().date().isoformat():
        guards["paused"] = True

    # Compute thresholds
    start_cash = float(day.get("start_cash", usd)) or usd
    cap_usd = DAILY_LOSS_CAP_USD
    cap_pct_usd = start_cash * (DAILY_LOSS_CAP_PCT/100.0)
    hard_cap = -min(cap_usd, cap_pct_usd)

    if day["pnl_today"] <= hard_cap and not guards["paused"]:
        print(f"{ts()} WARN: Guard DAILY_LOSS_CAP reached ({day['pnl_today']:.2f} ≤ {hard_cap:.2f}). Pausing buys for the day.")
        save_json(PAUSE_FILE, {"date": utcnow().date().isoformat(), "reason": "DAILY_LOSS_CAP"})
        guards["daily_cap"] = True
        guards["paused"] = True

    # Selection candidates (not held, not on cooldown)
    held_syms = {p.symbol for p in positions}
    now_ts = utcnow().timestamp()
    candidates = [s for s in ranked if s not in held_syms and cooldown.get(s, 0) <= now_ts]

    # Capacity & other guards
    cap_left = max(0, MAX_POSITIONS - len(positions))

    if guards["paused"]:
        pass  # already paused
    elif day.get("entries_today", 0) >= MAX_ENTRIES_PER_DAY:
        print(f"{ts()} WARN: Guard MAX_ENTRIES_PER_DAY hit ({day['entries_today']}/{MAX_ENTRIES_PER_DAY}).")
        guards["max_entries"] = True
    elif usd - RESERVE_USD < USD_PER_TRADE:
        print(f"{ts()} WARN: Guard RESERVE_USD active (reserve ${RESERVE_USD:.2f}). Skipping buys.")
        guards["reserve"] = True
    elif cap_left <= 0:
        print(f"{ts()} INFO: Capacity full (open={len(positions)}/{MAX_POSITIONS}). No buys this run.")
    elif not candidates:
        print(f"{ts()} INFO: No candidates (cooldown/held filtered). No buys this run.")
        guards["cooldown"] = True
    else:
        # Pick from top-K
        k = min(TOPK, len(candidates))
        top = candidates[:k]
        pick = None
        if PICK_MODE == "rotate":
            rot = load_json(ROTATE_FILE, {"idx": 0})
            idx = rot.get("idx", 0) % k
            pick = top[idx]
            rot["idx"] = (idx + 1) % k
            save_json(ROTATE_FILE, rot)
        else:  # random
            random.seed(utcnow().strftime("%Y%m%d%H"))  # hourly seed for mild stability
            pick = random.choice(top)

        # Execute buy
        px, qty = market_buy(ex, pick, USD_PER_TRADE)
        if qty > 0 and px > 0:
            pos = Position(pick, qty, px, TP_PCT, SL_PCT, TRAIL_PCT)
            positions.append(pos)
            day["entries_today"] = int(day.get("entries_today", 0)) + 1
            save_json(DAY_FILE, day)

    # Save positions state
    save_positions([p.to_d() for p in positions])

    # KPI SUMMARY
    guards_str = json.dumps(guards, separators=(",", ":"))
    cap_left = max(0, MAX_POSITIONS - len(positions))
    print(f"{ts()} INFO: KPI SUMMARY | dry_run={DRY_RUN} | open={len(positions)} | cap_left={cap_left} | usd=${usd:.2f} | pnl_today=${day['pnl_today']:.2f} | entries_today={day.get('entries_today',0)} | guards={guards_str}")

    append_history({
        "ts": utcnow().isoformat(),
        "dry_run": DRY_RUN,
        "open": len(positions),
        "cap_left": cap_left,
        "usd": usd,
        "pnl_today": day["pnl_today"],
        "entries_today": day.get("entries_today", 0),
        "guards": guards_str,
    })

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"{ts()} ERROR: Unhandled exception: {e}")
        raise
