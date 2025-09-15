# main.py
# Live/DRY crypto loop with:
#  - BUY + SELL (TP/SL)
#  - Persistent DAILY cap across runs (UTC) via state/daily.json
#  - Max open positions gate
#  - UNIVERSE from env (comma-separated), RSI gate, drop gate
#
# Works with GitHub Actions cache steps in crypto-live.yml.

import os, time, json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Optional

from trader.broker_crypto_ccxt import CCXTCryptoBroker


# -------------------- env helpers --------------------

def env_float(name: str, default: float) -> float:
    v = os.getenv(name, "")
    try:
        return float(v) if v != "" else float(default)
    except Exception:
        return float(default)

def env_int(name: str, default: int) -> int:
    v = os.getenv(name, "")
    try:
        return int(v) if v != "" else int(default)
    except Exception:
        return int(default)

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "")
    if v == "":
        return default
    return str(v).lower() in ("1", "true", "yes", "y", "on")

def parse_universe() -> Tuple[List[str], str]:
    default_universe = ["BTC/USD", "ETH/USD", "DOGE/USD"]
    raw = (os.getenv("UNIVERSE") or "").strip()
    if raw:
        u = [s.strip() for s in raw.split(",") if s.strip()]
        return u, "manual"
    return default_universe, "default"


# -------------------- indicators --------------------

def rsi_14(closes: List[float]) -> Optional[float]:
    period = 14
    if len(closes) < period + 1:
        return None
    gains = losses = 0.0
    for i in range(-period, 0):
        delta = closes[i] - closes[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses += -delta
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def pct_drop_from_prev(closes: List[float]) -> Optional[float]:
    if len(closes) < 2:
        return None
    prev, last = closes[-2], closes[-1]
    if prev == 0:
        return None
    return (last - prev) / prev * 100.0


# -------------------- config --------------------

DRY_RUN            = env_bool("DRY_RUN", True)
PRIVATE_API        = env_bool("PRIVATE_API", True)
PER_TRADE_USD      = env_float("PER_TRADE_USD", 10.0)
DAILY_CAP_USD      = env_float("DAILY_CAP_USD", 15.0)
DROP_GATE          = env_float("DROP_GATE", 0.60)          # need drop <= -DROP_GATE
TAKE_PROFIT_PCT    = env_float("TAKE_PROFIT_PCT", 3.0)     # SELL if pnl% >= TP
STOP_LOSS_PCT      = env_float("STOP_LOSS_PCT", 2.0)       # SELL if pnl% <= -SL
TRAIL_START_PCT    = env_float("TRAIL_START_PCT", 3.0)     # (reserved; not used here)
TRAIL_OFFSET_PCT   = env_float("TRAIL_OFFSET_PCT", 1.0)    # (reserved; not used here)
TF_MINUTES         = env_int("TF_MINUTES", 15)
RSI_MAX            = env_float("RSI_MAX", 60.0)            # loosened a bit
MAX_OPEN_POSITIONS = env_int("MAX_OPEN_POSITIONS", 3)      # cap concurrent holdings

EXCHANGE_ID        = os.getenv("EXCHANGE", "kraken")
UNIVERSE, UNIVERSE_MODE = parse_universe()

DUST_USD = 2.00  # display-only dust threshold


# -------------------- persistent daily cap --------------------

STATE_DIR  = Path("state")
STATE_FILE = STATE_DIR / "daily.json"

def utc_today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def load_daily_state() -> Dict[str, Any]:
    today = utc_today_str()
    try:
        data = json.loads(STATE_FILE.read_text())
        if data.get("date") == today and isinstance(data.get("spent_usd"), (int, float)):
            return {"date": today, "spent_usd": float(data["spent_usd"])}
    except Exception:
        pass
    return {"date": today, "spent_usd": 0.0}

def save_daily_state(state: Dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))

daily_state = load_daily_state()
# Ensure the path/file exists even if we make no buys this run
save_daily_state(daily_state)


# -------------------- connect --------------------

broker = CCXTCryptoBroker(
    exchange_id=EXCHANGE_ID,
    dry_run=DRY_RUN,
    enable_private=PRIVATE_API,
)
exchange = broker.exchange


# -------------------- banner --------------------

now = time.strftime('%Y-%m-%dT%H:%M:%S+00:00')
print("=== START TRADING OUTPUT ===")
print(
    f"{now} | run started | DRY_RUN={'True' if DRY_RUN else 'False'} | "
    f"TP={TAKE_PROFIT_PCT:.2f}% | SL={STOP_LOSS_PCT:.2f}% | "
    f"TRAIL_START={TRAIL_START_PCT:.2f}% | TRAIL_OFFSET={TRAIL_OFFSET_PCT:.2f}% | "
    f"DROP_GATE={DROP_GATE:.2f}% | TF={TF_MINUTES}m | RSI(14) ON max≤{RSI_MAX:.2f} | "
    f"private_api={'ON' if PRIVATE_API else 'OFF'}"
)
print(f"{now} | universe_mode={UNIVERSE_MODE}")
print(f"{now} | scanning={UNIVERSE}")


# -------------------- balance, positions & dust --------------------

def last_price(symbol: str) -> float:
    t = exchange.fetch_ticker(symbol)
    return float(t.get("last") or t.get("close") or 0.0)

def log_dust_and_count_positions(symbol: str, free_map: Dict[str, float]) -> Tuple[bool, bool]:
    """
    Returns (is_dust_printed, counts_as_position)
    """
    base = symbol.split("/")[0]
    qty = float(free_map.get(base, 0) or 0.0)
    lp = 0.0
    try:
        lp = last_price(symbol)
    except Exception:
        pass
    value = qty * lp if (qty and lp) else 0.0
    if value < DUST_USD:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | {symbol} | qty={qty:.8f} | last={lp:.2f} | value=${value:.2f} < dust(${DUST_USD:.2f}) -> ignore")
        return True, False
    else:
        return False, qty > 0

try:
    bal = exchange.fetch_balance() if PRIVATE_API else {"free": {}}
except Exception:
    bal = {"free": {}}
free_map = (bal.get("free") or {})

dust_ignored = 0
open_positions = 0
for sym in UNIVERSE:
    try:
        is_dust, counts_pos = log_dust_and_count_positions(sym, free_map)
        if is_dust:
            dust_ignored += 1
        if counts_pos:
            open_positions += 1
    except Exception:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | {sym} | dust-check error -> ignore")


# -------------------- budget (persistent) --------------------

usd_key, usd_free_amt = broker.get_free_cash(prefer=broker.USD_KEYS)
daily_spent = float(daily_state.get("spent_usd", 0.0))
daily_remaining = max(0.0, DAILY_CAP_USD - daily_spent)

print(
    f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | budget | USD_free=${usd_free_amt:.2f} | "
    f"daily_spent=${daily_spent:.2f} | daily_remaining=${daily_remaining:.2f} | "
    f"open_trades={open_positions}/{MAX_OPEN_POSITIONS} | dust_ignored={dust_ignored}"
)


# -------------------- SELL pass (TP/SL) --------------------

def estimate_entry_price(symbol: str, target_qty: float) -> Optional[float]:
    try:
        trades = exchange.fetch_my_trades(symbol, limit=200)
    except Exception:
        return None

    need = float(target_qty)
    if need <= 0:
        return None

    cost = 0.0
    got = 0.0
    trades_sorted = sorted(trades, key=lambda t: t.get("timestamp", 0), reverse=True)
    for tr in trades_sorted:
        if str(tr.get("side", "")).lower() != "buy":
            continue
        amt = float(tr.get("amount") or 0.0)
        price = float(tr.get("price") or 0.0)
        if amt <= 0 or price <= 0:
            continue
        take = min(amt, max(0.0, need - got))
        cost += take * price
        got += take
        if got >= need - 1e-12:
            break

    if got > 0:
        return cost / got
    return None

sells_placed = 0
for sym in UNIVERSE:
    base = sym.split("/")[0]
    pos_qty = float((free_map.get(base) or 0.0))
    if pos_qty <= 0:
        continue

    try:
        lp = last_price(sym)
    except Exception:
        continue

    if pos_qty * lp < DUST_USD:
        continue

    entry = estimate_entry_price(sym, pos_qty) if PRIVATE_API else None
    if entry is None or entry <= 0:
        continue

    pnl_pct = (lp - entry) / entry * 100.0
    reason = None
    if pnl_pct <= -abs(STOP_LOSS_PCT):
        reason = f"stop-loss {pnl_pct:.2f}% ≤ -{STOP_LOSS_PCT:.2f}%"
    elif pnl_pct >= abs(TAKE_PROFIT_PCT):
        reason = f"take-profit {pnl_pct:.2f}% ≥ {TAKE_PROFIT_PCT:.2f}%"

    if reason:
        try:
            if DRY_RUN:
                print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | DRY_RUN SELL {sym} all ({reason})")
            else:
                broker.market_sell_all(sym)
                print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | SELL placed {sym} all ({reason})")
                sells_placed += 1
        except Exception as e:
            print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | SELL error {sym}: {e}")


# -------------------- BUY scan --------------------

import ccxt  # for OHLCV
timeframe = f"{TF_MINUTES}m"
rows: List[Dict[str, Any]] = []
for sym in UNIVERSE:
    try:
        ohlcv = exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=50)
        closes = [c[4] for c in ohlcv if c and len(c) >= 5]
        if len(closes) < 15:
            continue
        rsi = rsi_14(closes)
        drop = pct_drop_from_prev(closes)  # negative = drop
        if rsi is None or drop is None:
            continue
        rows.append({"symbol": sym, "rsi": rsi, "drop": drop})
    except Exception:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | {sym} | ohlcv error -> skip")

rows_sorted = sorted(rows, key=lambda r: (r["drop"], r["rsi"]))  # more negative drop first

if rows_sorted:
    preview = rows_sorted[:5]
    preview_fmt = [f"{r['symbol']} Δ={r['drop']:.2f}% rsi={r['rsi']:.2f} {'x' if r['rsi']>RSI_MAX else ''}" for r in preview]
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | preview_top5 = [{'; '.join(preview_fmt)}]")

best = rows_sorted[0] if rows_sorted else None
will_buy = False
reason = ""
if best:
    drop_ok = (best["drop"] <= -abs(DROP_GATE))
    rsi_ok = (best["rsi"] <= RSI_MAX)
    pos_ok = (open_positions < MAX_OPEN_POSITIONS)
    cap_ok = (daily_remaining >= PER_TRADE_USD)
    cash_ok = (usd_free_amt >= PER_TRADE_USD)

    will_buy = drop_ok and rsi_ok and pos_ok and cap_ok and cash_ok
    if not will_buy:
        parts = []
        if not drop_ok: parts.append(f"drop {best['drop']:.2f}% > -{DROP_GATE:.2f}%")
        if not rsi_ok:  parts.append(f"RSI {best['rsi']:.2f} > {RSI_MAX:.2f}")
        if not pos_ok:  parts.append(f"positions {open_positions}/{MAX_OPEN_POSITIONS} full")
        if not cap_ok:  parts.append(f"daily_remaining ${daily_remaining:.2f} < ${PER_TRADE_USD:.2f}")
        if not cash_ok: parts.append(f"USD_free ${usd_free_amt:.2f} < ${PER_TRADE_USD:.2f}")
        reason = "; ".join(parts) if parts else "gates not met"
    else:
        reason = "passed gates"

buys_placed = 0
if not best:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | No viable candidates -> NO BUY")
elif not will_buy:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | Best candidate | {best['symbol']} ({reason}) -> NO BUY")
else:
    symbol = best["symbol"]
    try:
        if DRY_RUN:
            print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | DRY_RUN BUY {symbol} for ${PER_TRADE_USD:.2f} (drop={best['drop']:.2f}% rsi={best['rsi']:.2f})")
        else:
            broker.place_market_notional(symbol, PER_TRADE_USD)
            print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | BUY placed {symbol} for ~${PER_TRADE_USD:.2f} (drop={best['drop']:.2f}% rsi={best['rsi']:.2f})")
            buys_placed += 1
            # persist spend
            ds = {"date": utc_today_str(), "spent_usd": float(daily_state.get("spent_usd", 0.0)) + PER_TRADE_USD}
            save_daily_state(ds)
    except Exception as e:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | BUY error {symbol}: {e}")

# End-of-run: ensure the file exists (already saved above, but double-safe)
save_daily_state(load_daily_state())

print(f"Run complete. buys_placed={buys_placed} | sells_placed={sells_placed} | DRY_RUN={'True' if DRY_RUN else 'False'}")
print("=== END TRADING OUTPUT ===")
