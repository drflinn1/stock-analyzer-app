# main.py
# Minimal live/DRY trading loop using CCXT + Kraken-safe broker wrapper.
# - Reads UNIVERSE from env (comma-separated). If absent, uses a safe default.
# - Computes RSI(14) from ccxt OHLCV.
# - Chooses best candidate by largest negative % drop that also passes RSI cap.
# - Places a market buy sized by PER_TRADE_USD when gates pass and budget remains.
# - Prints clear logs matching your prior runs.

import os
import time
from typing import List, Tuple, Dict, Any, Optional

import ccxt  # used indirectly for OHLCV
from trader.broker_crypto_ccxt import CCXTCryptoBroker


# ---------- Helpers ---------- #

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

def rsi_14(closes: List[float]) -> Optional[float]:
    """
    Simple RSI(14) on a list of closes (>= 15).
    Returns None if not enough data.
    """
    period = 14
    if len(closes) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
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
    """
    % change of last close vs previous close (negative = drop).
    """
    if len(closes) < 2:
        return None
    prev = closes[-2]
    last = closes[-1]
    if prev == 0:
        return None
    return (last - prev) / prev * 100.0


# ---------- Config from env ---------- #

DRY_RUN            = env_bool("DRY_RUN", True)
PRIVATE_API        = env_bool("PRIVATE_API", True)
PER_TRADE_USD      = env_float("PER_TRADE_USD", 10.0)
DAILY_CAP_USD      = env_float("DAILY_CAP_USD", 15.0)
DROP_GATE          = env_float("DROP_GATE", 0.60)         # buy if drop <= -DROP_GATE
TAKE_PROFIT_PCT    = env_float("TAKE_PROFIT_PCT", 3.0)    # informational (not used yet)
STOP_LOSS_PCT      = env_float("STOP_LOSS_PCT", 2.0)      # informational (not used yet)
TRAIL_START_PCT    = env_float("TRAIL_START_PCT", 3.0)    # informational (not used yet)
TRAIL_OFFSET_PCT   = env_float("TRAIL_OFFSET_PCT", 1.0)   # informational (not used yet)
TF_MINUTES         = env_int("TF_MINUTES", 15)
RSI_MAX            = env_float("RSI_MAX", 35.0)           # cap like "RSI(14) ON max≤35.00"
EXCHANGE_ID        = os.getenv("EXCHANGE", "kraken")

UNIVERSE, UNIVERSE_MODE = parse_universe()
DUST_USD = 2.00  # ignore < $2 position prints

# ---------- Connect broker/exchange ---------- #

broker = CCXTCryptoBroker(
    exchange_id=EXCHANGE_ID,
    dry_run=DRY_RUN,
    enable_private=PRIVATE_API,
)

exchange = broker.exchange  # ccxt instance for OHLCV

# ---------- Banner ---------- #

print("=== START TRADING OUTPUT ===")
print(
    f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | run started | "
    f"DRY_RUN={'True' if DRY_RUN else 'False'} | "
    f"TP={TAKE_PROFIT_PCT:.2f}% | SL={STOP_LOSS_PCT:.2f}% | "
    f"TRAIL_START={TRAIL_START_PCT:.2f}% | TRAIL_OFFSET={TRAIL_OFFSET_PCT:.2f}% | "
    f"DROP_GATE={DROP_GATE:.2f}% | TF={TF_MINUTES}m | "
    f"RSI(14) ON max≤{RSI_MAX:.2f} | private_api={'ON' if PRIVATE_API else 'OFF'}"
)
print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | universe_mode={UNIVERSE_MODE}")
print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | scanning={UNIVERSE}")

# ---------- Print tiny holdings (dust) like your logs ---------- #

try:
    bal = exchange.fetch_balance() if PRIVATE_API else {"free": {}}
except Exception:
    bal = {"free": {}}
free_map = (bal.get("free") or {})

def log_dust(symbol: str):
    base = symbol.split("/")[0]
    qty = float(free_map.get(base, 0) or 0.0)
    last_price = None
    try:
        t = exchange.fetch_ticker(symbol)
        last_price = float(t.get("last") or t.get("close") or 0.0)
    except Exception:
        last_price = 0.0
    value = qty * last_price if (qty and last_price) else 0.0
    if value < DUST_USD:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | {symbol} | qty={qty:.8f} | last={last_price:.2f} | value=${value:.2f} < dust(${DUST_USD:.2f}) -> ignore")
        return True
    return False

dust_ignored = 0
for sym in UNIVERSE:
    try:
        if log_dust(sym):
            dust_ignored += 1
    except Exception:
        # log but keep going
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | {sym} | dust-check error -> ignore")

# ---------- Budget / open-trade bookkeeping (simple) ---------- #

open_trades = 0  # if you track positions elsewhere, wire it here
daily_remaining = DAILY_CAP_USD

usd_key, usd_free_amt = broker.get_free_cash(prefer=broker.USD_KEYS)
print(
    f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | budget | USD_free=${usd_free_amt:.2f} | "
    f"daily_remaining=${daily_remaining:.2f} | open_trades={open_trades}/3 | dust_ignored={dust_ignored}"
)

# ---------- Scan universe: compute RSI and last 1-candle % drop ---------- #

timeframe = f"{TF_MINUTES}m"
rows: List[Dict[str, Any]] = []
for sym in UNIVERSE:
    try:
        ohlcv = exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=50)
        closes = [c[4] for c in ohlcv if c and len(c) >= 5]
        if len(closes) < 15:
            continue
        rsi = rsi_14(closes)
        drop = pct_drop_from_prev(closes)  # negative means drop
        rows.append({"symbol": sym, "rsi": rsi, "drop": drop})
    except Exception as e:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | {sym} | ohlcv error -> skip")

# Rank candidates: larger negative drop first, then lower RSI
rows = [
    r for r in rows
    if r["rsi"] is not None and r["drop"] is not None
]
rows_sorted = sorted(rows, key=lambda r: (r["drop"], r["rsi"]))  # drop ascending (more negative first)

# preview_top5
preview = rows_sorted[:5]
if preview:
    preview_fmt = []
    for r in preview:
        preview_fmt.append(f"{r['symbol']} Δ={r['drop']:.2f}% rsi={r['rsi']:.2f} {'x' if r['rsi']>RSI_MAX else ''}")
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | preview_top5 = [{'; '.join(preview_fmt)}]")

# Pick best candidate that meets gates
best: Optional[Dict[str, Any]] = rows_sorted[0] if rows_sorted else None
reason = ""
will_buy = False
if best:
    drop_ok = (best["drop"] <= -abs(DROP_GATE))
    rsi_ok = (best["rsi"] <= RSI_MAX)
    if not drop_ok or not rsi_ok:
        reason = f"did not pass: drop {best['drop']:.2f}% {'<' if drop_ok else '>'} gate, RSI {best['rsi']:.2f} {'≤' if rsi_ok else '>'} {RSI_MAX:.2f}"
    else:
        reason = "passed gates"
        will_buy = True

# ---------- Execute buy if allowed ---------- #

buys_placed = 0
sells_placed = 0

if not will_buy:
    if best:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | Best candidate | {best['symbol']} ({reason}) -> NO BUY")
    else:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | No viable candidates -> NO BUY")
else:
    symbol = best["symbol"]
    # gates: budget and cash
    if daily_remaining < PER_TRADE_USD:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | BUY window skipped (budget ${daily_remaining:.2f} < per-trade ${PER_TRADE_USD:.2f})")
    elif usd_free_amt < PER_TRADE_USD:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | BUY window skipped (insufficient USD: ${usd_free_amt:.2f})")
    else:
        try:
            if DRY_RUN:
                # Simulate order
                print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | DRY_RUN BUY {symbol} for ${PER_TRADE_USD:.2f} (drop={best['drop']:.2f}% rsi={best['rsi']:.2f})")
            else:
                broker.place_market_notional(symbol, PER_TRADE_USD)
                print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | BUY placed {symbol} for ~${PER_TRADE_USD:.2f} (drop={best['drop']:.2f}% rsi={best['rsi']:.2f})")
                buys_placed += 1
                daily_remaining = max(0.0, daily_remaining - PER_TRADE_USD)
        except Exception as e:
            print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | BUY error {symbol}: {e}")

# (Future) Sell/TP/SL/Trailing logic would go here.

print(f"Run complete. buys_placed={buys_placed} | sells_placed={sells_placed} | DRY_RUN={'True' if DRY_RUN else 'False'}")
print("=== END TRADING OUTPUT ===")
