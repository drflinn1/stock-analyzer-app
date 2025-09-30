# main.py — Crypto Live with Guard Pack
# - USD-only whitelist (BTC/ETH/SOL/DOGE)
# - Auto-pick Top-K with cooldown
# - Take-profit / Stop-loss / Trailing stop
# - Reserve cash, daily loss cap, max daily entries
# - Emergency SL, auto-pause guard, KPI/CSV, DRY-RUN banner

from __future__ import annotations
import os, json, random, csv
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

# ---------- ENV & DEFAULTS ---------- #

def _as_bool(val: Optional[str], default: bool) -> bool:
    if val is None: return default
    return str(val).strip().lower() in {"1","true","yes","on"}

def _as_float(val: Optional[str], default: float) -> float:
    try:
        return float(val) if val is not None and str(val).strip() != "" else default
    except: return default

def _as_int(val: Optional[str], default: int) -> int:
    try:
        return int(float(val)) if val is not None and str(val).strip() != "" else default
    except: return default

DRY_RUN        = _as_bool(os.getenv("DRY_RUN"), True)
RUN_SWITCH     = os.getenv("RUN_SWITCH", "ON").upper()  # "ON" or "OFF"
USD_PER_TRADE  = _as_float(os.getenv("USD_PER_TRADE"), 10.0)
MAX_POSITIONS  = _as_int(os.getenv("MAX_POSITIONS"), 3)

# Balanced exits by default
TP_PCT         = _as_float(os.getenv("TP_PCT"), 3.5)   # %
SL_PCT         = _as_float(os.getenv("SL_PCT"), 2.0)   # %
TRAIL_PCT      = _as_float(os.getenv("TRAIL_PCT"), 1.2) # %

# Guard Pack
RESERVE_USD         = _as_float(os.getenv("RESERVE_USD"), 100.0)
DAILY_LOSS_CAP_USD  = _as_float(os.getenv("DAILY_LOSS_CAP_USD"), 5.0)
DAILY_LOSS_CAP_PCT  = _as_float(os.getenv("DAILY_LOSS_CAP_PCT"), 2.0)  # % of start cash
MAX_ENTRIES_PER_DAY = _as_int(os.getenv("MAX_ENTRIES_PER_DAY"), 6)
COOLDOWN_MIN        = _as_int(os.getenv("COOLDOWN_MIN"), 120)  # ~8 runs at 15m
TOPK                = max(1, _as_int(os.getenv("TOPK"), 3))
PICK_MODE           = os.getenv("PICK_MODE", "random").lower()  # "random" | "rotate"
EMERGENCY_SL_PCT    = _as_float(os.getenv("EMERGENCY_SL_PCT"), 3.5)

STATE_DIR      = ".state"
POSITIONS_FILE = os.path.join(STATE_DIR, "positions.json")
DAY_FILE       = os.path.join(STATE_DIR, "day_state.json")
COOLDOWN_FILE  = os.path.join(STATE_DIR, "cooldown.json")
PAUSE_FILE     = os.path.join(STATE_DIR, "guard_pause.json")
HISTORY_CSV    = os.path.join(STATE_DIR, "kpi_history.csv")
ROTATE_FILE    = os.path.join(STATE_DIR, "rotate_idx.json")

USD_KEYS  = ("USD", "ZUSD")
UNIVERSE  = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]

os.makedirs(STATE_DIR, exist_ok=True)

# ---------- UTIL ---------- #

def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default

def _save_json(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def _append_history(row: Dict[str, Any]):
    header = ["ts","dry_run","open","cap_left","usd","pnl_today","entries_today","guards"]
    newfile = not os.path.exists(HISTORY_CSV)
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if newfile: w.writeheader()
        w.writerow({k: row.get(k) for k in header})

# ---------- EXCHANGE ---------- #

def _get_exchange() -> Any:
    api_key = os.getenv("KRAKEN_API_KEY")
    api_secret = os.getenv("KRAKEN_API_SECRET")
    cfg = {"enableRateLimit": True}
    if not DRY_RUN and api_key and api_secret:
        cfg.update({"apiKey": api_key, "secret": api_secret})
    return ccxt.kraken(cfg)

# ---------- BALANCES ---------- #

def _get_usd_balance(ex) -> float:
    bal = ex.fetch_balance({})
    total = 0.0
    for k in USD_KEYS:
        if k in bal.get("total", {}):
            total += float(bal["total"][k] or 0)
        elif k in bal:
            total += float(bal.get(k) or 0)
    return float(total)

# ---------- PRICES / RANKING ---------- #

def _last_price(ex, symbol: str) -> Optional[float]:
    try:
        t = ex.fetch_ticker(symbol)
        px = t.get("last") or t.get("close") or t.get("bid") or t.get("ask")
        return float(px) if px else None
    except Exception:
        return None

def _momentum_score(ex, symbol: str, tf: str = "15m", lookback: int = 12) -> float:
    """Simple 15m momentum over ~3 hours. Fallback to 0 on failure."""
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, limit=lookback)
        if not ohlcv or len(ohlcv) < 2:
            return 0.0
        first = float(ohlcv[0][4]); last = float(ohlcv[-1][4])
        return (last - first) / first
    except Exception:
        return 0.0

# ---------- STATE ---------- #

def _load_positions() -> List[Dict[str, Any]]:
    return _load_json(POSITIONS_FILE, [])

def _save_positions(positions: List[Dict[str, Any]]):
    _save_json(POSITIONS_FILE, positions)

def _load_cooldown() -> Dict[str, float]:
    data = _load_json(COOLDOWN_FILE, {})
    now = utcnow().timestamp()
    data = {sym: exp for sym, exp in data.items() if exp > now}  # prune
    _save_json(COOLDOWN_FILE, data)
    return data

def _set_cooldown(sym: str):
    data = _load_json(COOLDOWN_FILE, {})
    exp = (utcnow() + timedelta(minutes=COOLDOWN_MIN)).timestamp()
    data[sym] = exp
    _save_json(COOLDOWN_FILE, data)

def _load_day_state(now_usd: float) -> Dict[str, Any]:
    today = utcnow().date().isoformat()
    ds = _load_json(DAY_FILE, {})
    if ds.get("date") != today:
        ds = {"date": today, "start_cash": now_usd, "pnl_today": 0.0, "entries_today": 0}
        if os.path.exists(PAUSE_FILE):
            try: os.remove(PAUSE_FILE)
            except: pass
    _save_json(DAY_FILE, ds)
    return ds

# ---------- BROKER OPS (SIM / LIVE) ---------- #

def _market_buy(ex, symbol: str, usd: float) -> Tuple[float, float]:
    px = _last_price(ex, symbol) or 0.0
    amt = 0.0 if px <= 0 else usd / px
    if DRY_RUN:
        print(f"{ts()} INFO: [DRY RUN] ccxt BUY {symbol} amount={amt:.8f} (~${usd:.2f}) @~{px}")
        return px, amt
    try:
        order = ex.create_market_buy_order(symbol, amt)
        px2 = float(order.get("average") or px)
        amt2 = float(order.get("amount") or amt)
        print(f"{ts()} INFO: LIVE BUY executed: {symbol} ${usd:.2f} avg~{px2}")
        return px2, amt2
    except Exception as e:
        print(f"{ts()} ERROR: BUY failed: {e}")
        return px, amt

def _market_sell(ex, symbol: str, qty: float) -> float:
    px = _last_price(ex, symbol) or 0.0
    if DRY_RUN:
        print(f"{ts()} INFO: [DRY RUN] ccxt SELL {symbol} qty={qty:.8f}")
        print(f"{ts()} INFO: SELL executed: {symbol} qty={qty:.8f}")
        return px
    try:
        order = ex.create_market_sell_order(symbol, qty)
        px2 = float(order.get("average") or px)
        print(f"{ts()} INFO: LIVE SELL executed: {symbol} qty={qty:.8f} avg~{px2}")
        return px2
    except Exception as e:
        print(f"{ts()} ERROR: SELL failed: {e}")
        return px

# ---------- POSITION ---------- #

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
        p = Position(d["symbol"], float(d["qty"]), float(d["entry"]),
                     float(d["tp_pct"]), float(d["sl_pct"]), float(d["trail_pct"]))
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
        if price > self.high:
            self.high = price
        change = (price - self.entry) / self.entry * 100.0
        trail_stop = self.high * (1.0 - self.trail_pct/100.0)

        # Order: emergency, TP, SL, trail
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

# ---------- MAIN ---------- #

def main():
    print("=========================================================")
    print("🚧 DRY RUN — NO REAL ORDERS SENT 🚧" if DRY_RUN else "⚠️ LIVE MODE — REAL ORDERS ENABLED ⚠️")
    print("=========================================================")

    if RUN_SWITCH == "OFF":
        print(f"{ts()} WARN: RUN_SWITCH=OFF — exiting early.")
        return

    ex = _get_exchange()
    print(f"{ts()} INFO: Starting trader in CRYPTO mode. Dry run={DRY_RUN}. Broker=ccxt")
    print(f"{ts()} INFO: Using crypto broker class: CryptoBroker")

    # Balances
    try:
        usd = _get_usd_balance(ex)
    except Exception as e:
        print(f"{ts()} ERROR: Balance fetch failed: {e}")
        usd = 0.0
    print(f"{ts()} INFO: [ccxt] USD balance detected: ${usd:.2f}")
    print(f"{ts()} INFO: Balance via get_balance: ${usd:.2f}")
    print(f"{ts()} INFO: Available balance: ${usd:.2f}")

    # State
    day = _load_day_state(usd)
    positions = [Position.from_d(p) for p in _load_positions()]
    cooldown = _load_cooldown()

    # Rank universe
    scored = []
    for sym in UNIVERSE:
        scored.append((sym, _momentum_score(ex, sym, tf="15m", lookback=12)))
    scored.sort(key=lambda x: x[1], reverse=True)
    ranked = [s for s,_ in scored]
    print(f"{ts()} INFO: Universe: {ranked}")

    # Exits for open positions
    realized_pnl = 0.0
    still_open: List[Position] = []
    for pos in positions:
        px = _last_price(ex, pos.symbol) or pos.entry
        reason = pos.check_exits(px)
        if reason:
            print(f"{ts()} INFO: {reason}")
            sell_px = _market_sell(ex, pos.symbol, pos.qty)
            pnl = (sell_px - pos.entry) * pos.qty
            realized_pnl += pnl
            _set_cooldown(pos.symbol)
        else:
            still_open.append(pos)
    positions = still_open

    # Update day PnL
    day["pnl_today"] = round(float(day.get("pnl_today", 0.0)) + realized_pnl, 2)
    _save_json(DAY_FILE, day)

    # ----- GUARDS -----
    guards = {"reserve": False, "daily_cap": False, "cooldown": False, "max_entries": False, "paused": False}

    pause = _load_json(PAUSE_FILE, {})
    if pause.get("date") == utcnow().date().isoformat():
        guards["paused"] = True

    start_cash = float(day.get("start_cash", usd)) or usd
    cap_usd = DAILY_LOSS_CAP_USD
    cap_pct_usd = start_cash * (DAILY_LOSS_CAP_PCT/100.0)
    hard_cap = -min(cap_usd, cap_pct_usd)

    if day["pnl_today"] <= hard_cap and not guards["paused"]:
        print(f"{ts()} WARN: Guard DAILY_LOSS_CAP reached ({day['pnl_today']:.2f} ≤ {hard_cap:.2f}). Pausing buys for the day.")
        _save_json(PAUSE_FILE, {"date": utcnow().date().isoformat(), "reason": "DAILY_LOSS_CAP"})
        guards["daily_cap"] = True
        guards["paused"] = True

    held_syms = {p.symbol for p in positions}
    now_ts = utcnow().timestamp()
    candidates = [s for s in ranked if s not in held_syms and _load_json(COOLDOWN_FILE, {}).get(s, 0) <= now_ts]
    cap_left = max(0, MAX_POSITIONS - len(positions))

    if guards["paused"]:
        pass
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
        if PICK_MODE == "rotate":
            rot = _load_json(ROTATE_FILE, {"idx": 0}); idx = rot.get("idx", 0) % k
            pick = top[idx]; rot["idx"] = (idx + 1) % k; _save_json(ROTATE_FILE, rot)
        else:
            random.seed(utcnow().strftime("%Y%m%d%H"))
            pick = random.choice(top)

        px, qty = _market_buy(ex, pick, USD_PER_TRADE)
        if qty > 0 and px > 0:
            positions.append(Position(pick, qty, px, TP_PCT, SL_PCT, TRAIL_PCT))
            day["entries_today"] = int(day.get("entries_today", 0)) + 1
            _save_json(DAY_FILE, day)

    # Save positions
    _save_positions([p.to_d() for p in positions])

    # KPI SUMMARY
    guards_str = json.dumps(guards, separators=(",", ":"))
    cap_left = max(0, MAX_POSITIONS - len(positions))
    print(f"{ts()} INFO: KPI SUMMARY | dry_run={DRY_RUN} | open={len(positions)} | cap_left={cap_left} | usd=${usd:.2f} | pnl_today=${day['pnl_today']:.2f} | entries_today={day.get('entries_today',0)} | guards={guards_str}")
    _append_history({"ts": utcnow().isoformat(), "dry_run": DRY_RUN, "open": len(positions),
                     "cap_left": cap_left, "usd": usd, "pnl_today": day["pnl_today"],
                     "entries_today": day.get("entries_today", 0), "guards": guards_str})

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"{ts()} ERROR: Unhandled exception: {e}")
        raise
