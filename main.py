# main.py
# Live/DRY crypto loop with BUY + SELL (TP/SL) using CCXT on Kraken.
# - Reads UNIVERSE from env.
# - Sells positions if PnL <= -STOP_LOSS_PCT or PnL >= TAKE_PROFIT_PCT.
# - Buys best candidate by drop% + RSI gate, honoring per-trade/daily caps.

import os
import time
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
RSI_MAX            = env_float("RSI_MAX", 60.0)            # loosened so buys can happen tonight
EXCHANGE_ID        = os.getenv("EXCHANGE", "kraken")

UNIVERSE, UNIVERSE_MODE = parse_universe()
DUST_USD = 2.00  # display-only dust threshold

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

# -------------------- balance & dust print --------------------

def last_price(symbol: str) -> float:
    t = exchange.fetch_ticker(symbol)
    return float(t.get("last") or t.get("close") or 0.0)

def log_dust(symbol: str, free_map: Dict[str, float]) -> bool:
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
        return True
    return False

try:
    bal = exchange.fetch_balance() if PRIVATE_API else {"free": {}}
except Exception:
    bal = {"free": {}}
free_map = (bal.get("free") or {})
dust_ignored = 0
for sym in UNIVERSE:
    try:
        if log_dust(sym, free_map):
            dust_ignored += 1
    except Exception:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | {sym} | dust-check error -> ignore")

# -------------------- budget --------------------

open_trades = 0  # placeholder; wire to your position tracking if needed
daily_remaining = DAILY_CAP_USD
usd_key, usd_free_amt = broker.get_free_cash(prefer=broker.USD_KEYS)
print(
    f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | budget | USD_free=${usd_free_amt:.2f} | "
    f"daily_remaining=${daily_remaining:.2f} | open_trades={open_trades}/3 | dust_ignored={dust_ignored}"
)

# -------------------- SELL pass (TP/SL) --------------------
# Estimate entry price from trade history and compute PnL% for each held symbol.

def estimate_entry_price(symbol: str, target_qty: float) -> Optional[float]:
    """
    Estimate average entry for the current free quantity by walking back your trade
    history (most recent first) and summing buy trades until we cover target_qty.
    Returns VWAP entry price or None if not enough history.
    """
    try:
        trades = exchange.fetch_my_trades(symbol, limit=200)
    except Exception:
        return None

    need = float(target_qty)
    if need <= 0:
        return None

    cost = 0.0
    got = 0.0

    # oldest->newest to build in chronological order, but we only need buys
    trades_sorted = sorted(trades, key=lambda t: t.get("timestamp", 0), reverse=True)
    for tr in trades_sorted:
        if str(tr.get("side", "")).lower() != "buy":
            # sells reduce position; skip for entry calc (we aim to cover current free qty)
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

    # Ignore dust in sell logic too
    if pos_qty * lp < DUST_USD:
        continue

    entry = estimate_entry_price(sym, pos_qty) if PRIVATE_API else None
    if entry is None or entry <= 0:
        # No reliable entry; skip (we can bootstrap later)
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

# Rank: more negative drop first, then lower RSI
rows_sorted = sorted(rows, key=lambda r: (r["drop"], r["rsi"]))

# preview_top5
if rows_sorted:
    preview = rows_sorted[:5]
    preview_fmt = [f"{r['symbol']} Δ={r['drop']:.2f}% rsi={r['rsi']:.2f} {'x' if r['rsi']>RSI_MAX else ''}" for r in preview]
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | preview_top5 = [{'; '.join(preview_fmt)}]")

# choose best meeting gates
best = rows_sorted[0] if rows_sorted else None
will_buy = False
reason = ""
if best:
    drop_ok = (best["drop"] <= -abs(DROP_GATE))
    rsi_ok = (best["rsi"] <= RSI_MAX)
    if drop_ok and rsi_ok:
        will_buy = True
        reason = "passed gates"
    else:
        reason = f"did not pass: drop {best['drop']:.2f}% {'≤' if drop_ok else '>'} -{DROP_GATE:.2f}% gate, RSI {best['rsi']:.2f} {'≤' if rsi_ok else '>'} {RSI_MAX:.2f}"

buys_placed = 0
if not best:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | No viable candidates -> NO BUY")
elif not will_buy:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | Best candidate | {best['symbol']} ({reason}) -> NO BUY")
else:
    symbol = best["symbol"]
    if daily_remaining < PER_TRADE_USD:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | BUY window skipped (budget ${daily_remaining:.2f} < per-trade ${PER_TRADE_USD:.2f})")
    elif usd_free_amt < PER_TRADE_USD:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | BUY window skipped (insufficient USD: ${usd_free_amt:.2f})")
    else:
        try:
            if DRY_RUN:
                print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | DRY_RUN BUY {symbol} for ${PER_TRADE_USD:.2f} (drop={best['drop']:.2f}% rsi={best['rsi']:.2f})")
            else:
                broker.place_market_notional(symbol, PER_TRADE_USD)
                print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | BUY placed {symbol} for ~${PER_TRADE_USD:.2f} (drop={best['drop']:.2f}% rsi={best['rsi']:.2f})")
                buys_placed += 1
                daily_remaining = max(0.0, daily_remaining - PER_TRADE_USD)
        except Exception as e:
            print(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00')} | BUY error {symbol}: {e}")

print(f"Run complete. buys_placed={buys_placed} | sells_placed={sells_placed} | DRY_RUN={'True' if DRY_RUN else 'False'}")
print("=== END TRADING OUTPUT ===")
