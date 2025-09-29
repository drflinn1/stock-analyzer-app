# trader/crypto_engine.py
# Kraken spot (via CCXT) — USD-only pairs, DRY RUN with loud logs,
# simple entries + 3.5% TP, 2.0% SL, ATR-based trailing after +1.0%.

import os, json, math, time, pathlib, sys
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import ccxt

# ========= configurable (envs override) =========
STATE_DIR            = os.getenv("STATE_DIR", ".state")
POSITIONS_FILE       = os.getenv("POSITIONS_FILE", f"{STATE_DIR}/positions.json")
TIMEFRAME            = os.getenv("TIMEFRAME", "15m")
ATR_PERIOD           = int(os.getenv("ATR_PERIOD", "14"))
USD_PER_TRADE        = float(os.getenv("USD_PER_TRADE", "10"))
MAX_POSITIONS        = int(os.getenv("MAX_POSITIONS", "3"))

TAKE_PROFIT_PCT      = float(os.getenv("TAKE_PROFIT_PCT", "0.035"))  # +3.5%
STOP_LOSS_PCT        = float(os.getenv("STOP_LOSS_PCT",   "0.020"))  # -2.0%
TRAIL_ACTIVATE_PCT   = float(os.getenv("TRAIL_ACTIVATE_PCT", "0.010"))  # +1.0%
TRAIL_ATR_MULT       = float(os.getenv("TRAIL_ATR_MULT", "0.8"))

EXCHANGE_ID          = os.getenv("EXCHANGE_ID", "kraken")
DRY_RUN              = os.getenv("DRY_RUN", "true").lower() == "true"

# Optional fixed universe (comma-separated). If empty → auto USD-only from markets.
UNIVERSE_CSV         = os.getenv("UNIVERSE", "BTC/USD,ETH/USD,SOL/USD,XRP/USD,DOGE/USD")
USD_KEYS             = {"USD", "ZUSD"}
STABLES              = {"USDT", "USDC", "DAI"}

# ===============================================

def now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def log(msg: str, level: str = "INFO"):
    # Loud DRY RUN banner once per run
    if level == "BANNER":
        print("="*90)
        print(msg)
        print("="*90)
        return
    print(f"{now()} {level}: {msg}")

def ensure_dirs():
    pathlib.Path(STATE_DIR).mkdir(parents=True, exist_ok=True)

def load_positions() -> Dict[str, Any]:
    if not pathlib.Path(POSITIONS_FILE).exists():
        return {}
    try:
        with open(POSITIONS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_positions(data: Dict[str, Any]):
    ensure_dirs()
    with open(POSITIONS_FILE, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

def connect_exchange() -> ccxt.Exchange:
    exchange = getattr(ccxt, EXCHANGE_ID)({
        "apiKey": os.getenv("KRAKEN_API_KEY"),
        "secret": os.getenv("KRAKEN_API_SECRET"),
        "password": os.getenv("KRAKEN_API_PASSWORD") or None,
        "enableRateLimit": True,
    })
    exchange.load_markets()
    return exchange

def usd_balance(exchange: ccxt.Exchange) -> float:
    try:
        bal = exchange.fetch_balance()
        total = bal.get("total", {}) or {}
        return sum(float(total.get(k, 0)) for k in USD_KEYS)
    except Exception as e:
        log(f"Could not fetch USD/ZUSD balance: {e}", "WARN")
        return 0.0

def usd_only_universe(exchange: ccxt.Exchange) -> List[str]:
    markets = exchange.markets
    syms = []
    for sym, m in markets.items():
        if not m.get("active"):
            continue
        if m.get("quote") != "USD":
            continue
        base = m.get("base", "")
        if base in STABLES:   # skip stables
            continue
        syms.append(sym)
    if UNIVERSE_CSV:
        # keep order preference, and filter to available/active
        preferred = [s.strip() for s in UNIVERSE_CSV.split(",") if s.strip()]
        return [s for s in preferred if s in syms]
    return sorted(syms)

def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, limit: int = 200) -> List[List[float]]:
    # candles: [ts, open, high, low, close, volume]
    return exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)

def calc_atr(ohlcv: List[List[float]], period: int = 14) -> float:
    if len(ohlcv) < period + 1:
        return 0.0
    trs: List[float] = []
    for i in range(1, len(ohlcv)):
        _, _, high, low, close_prev, _ = ohlcv[i-1]
        _, _, h, l, c, _ = ohlcv[i]
        tr = max(h - l, abs(h - close_prev), abs(l - close_prev))
        trs.append(tr)
    trs = trs[-period:]
    if not trs:
        return 0.0
    return sum(trs) / len(trs)

def last_price(exchange: ccxt.Exchange, symbol: str) -> float:
    ticker = exchange.fetch_ticker(symbol)
    return float(ticker["last"])

def pick_entry_candidates(exchange: ccxt.Exchange, universe: List[str]) -> List[Tuple[str, float]]:
    """
    Very simple momentum proxy: close-to-EMA drift using last 50 vs 200 closes.
    Returns list of (symbol, score) sorted desc.
    """
    picks = []
    for sym in universe[:12]:  # cap requests
        try:
            ohlcv = fetch_ohlcv(exchange, sym, limit=250)
            closes = [c[4] for c in ohlcv]
            if len(closes) < 210:
                continue
            ema_fast = ema(closes, 50)[-1]
            ema_slow = ema(closes, 200)[-1]
            score = (ema_fast / ema_slow) - 1.0  # >0 = trending up
            picks.append((sym, score))
            time.sleep(exchange.rateLimit/1000)
        except Exception as e:
            log(f"{sym}: failed to fetch candles: {e}", "WARN")
    picks.sort(key=lambda x: x[1], reverse=True)
    return picks

def ema(series: List[float], period: int) -> List[float]:
    k = 2 / (period + 1)
    out: List[float] = []
    ema_val = series[0]
    for v in series:
        ema_val = v * k + ema_val * (1 - k)
        out.append(ema_val)
    return out

def amount_to_precision(exchange: ccxt.Exchange, symbol: str, amount: float) -> float:
    s = exchange.amount_to_precision(symbol, amount)
    return float(s)

# ===================== core =====================

def buy(exchange: ccxt.Exchange, symbol: str, usd_amount: float, positions: Dict[str, Any]):
    px = last_price(exchange, symbol)
    qty = usd_amount / px
    qty = amount_to_precision(exchange, symbol, qty)
    if DRY_RUN:
        log(f"🚧 DRY RUN — SIMULATED BUY {symbol} qty={qty} @~{px}", "INFO")
    else:
        log(f"LIVE BUY {symbol} qty={qty} (market) @~{px}", "INFO")
        exchange.create_order(symbol, "market", "buy", qty)
    positions[symbol] = {
        "symbol": symbol,
        "qty": qty,
        "entry": px,
        "trail_active": False,
        "trail_stop": None,
        "created": now(),
    }
    save_positions(positions)

def sell_all(exchange: ccxt.Exchange, symbol: str, reason: str, positions: Dict[str, Any]):
    pos = positions.get(symbol)
    if not pos:
        return
    qty = pos["qty"]
    qty = amount_to_precision(exchange, symbol, qty)
    px = last_price(exchange, symbol)
    if DRY_RUN:
        log(f"🚧 DRY RUN — SIMULATED SELL {symbol} qty={qty} @~{px}  reason={reason}", "INFO")
    else:
        log(f"LIVE SELL {symbol} qty={qty} (market) @~{px}  reason={reason}", "INFO")
        exchange.create_order(symbol, "market", "sell", qty)
    positions.pop(symbol, None)
    save_positions(positions)

def manage_position(exchange: ccxt.Exchange, symbol: str, positions: Dict[str, Any]):
    pos = positions[symbol]
    px = last_price(exchange, symbol)

    entry = pos["entry"]
    up_pct = (px / entry) - 1.0
    down_pct = 1.0 - (px / entry)

    # Take profit
    if up_pct >= TAKE_PROFIT_PCT:
        sell_all(exchange, symbol, f"TAKE_PROFIT hit at {TAKE_PROFIT_PCT*100:.2f}%", positions)
        return

    # Stop loss
    if down_pct >= STOP_LOSS_PCT:
        sell_all(exchange, symbol, f"STOP_LOSS hit at {STOP_LOSS_PCT*100:.2f}%", positions)
        return

    # Trailing logic (ATR-based once activated)
    if (not pos["trail_active"]) and up_pct >= TRAIL_ACTIVATE_PCT:
        # activate trail
        atr = calc_atr(fetch_ohlcv(exchange, symbol, limit=max(ATR_PERIOD+2, 100)), ATR_PERIOD)
        trail = px - (TRAIL_ATR_MULT * atr)
        pos["trail_active"] = True
        pos["trail_stop"]   = trail
        log(f"{symbol} TRAIL activated: price~{px:.6f} | ATR={atr:.6f} | stop~{trail:.6f}", "INFO")
        save_positions(positions)
        return

    if pos["trail_active"]:
        atr = calc_atr(fetch_ohlcv(exchange, symbol, limit=max(ATR_PERIOD+2, 100)), ATR_PERIOD)
        new_trail = px - (TRAIL_ATR_MULT * atr)
        # Trail stop ratchets up only
        if new_trail > (pos["trail_stop"] or -math.inf):
            pos["trail_stop"] = new_trail
            save_positions(positions)
        if px <= pos["trail_stop"]:
            sell_all(exchange, symbol, f"TRAIL_STOP {pos['trail_stop']:.6f} breached (px~{px:.6f})", positions)
            return

def run():
    ensure_dirs()

    # Loud banner about mode
    if DRY_RUN:
        log("🚧 DRY RUN — NO REAL ORDERS SENT 🚧", "BANNER")
    else:
        log("⚠️ LIVE MODE — REAL ORDERS WILL BE SENT ⚠️", "BANNER")

    log(f"Starting trader in CRYPTO mode. Dry run={DRY_RUN}. Broker=ccxt")
    exchange = connect_exchange()

    # Universe
    universe = usd_only_universe(exchange)
    log(f"Universe (USD-only): {universe}")

    # Balances
    usd_bal = usd_balance(exchange)
    log(f"[ccxt] USD/ZUSD balance detected: ${usd_bal:.2f}")

    # Load positions
    positions = load_positions()
    open_syms = list(positions.keys())
    log(f"Open positions: {open_syms if open_syms else 'none'}")

    # 1) Manage existing positions
    for sym in list(open_syms):
        try:
            manage_position(exchange, sym, positions)
            time.sleep(exchange.rateLimit/1000)
        except Exception as e:
            log(f"{sym} manage error: {e}", "WARN")

    # 2) Entry: simple momentum pick if capacity left
    positions = load_positions()  # reload (may have sold)
    cap_left = MAX_POSITIONS - len(positions)
    if cap_left > 0 and USD_PER_TRADE > 0:
        # find candidates not already held
        candidates = [s for s in universe if s not in positions]
        if candidates:
            ranked = pick_entry_candidates(exchange, candidates)
            if ranked:
                pick = ranked[0][0]
                # buy only if we have at least USD_PER_TRADE available
                if usd_bal >= USD_PER_TRADE or DRY_RUN:
                    try:
                        buy(exchange, pick, USD_PER_TRADE, positions)
                    except Exception as e:
                        log(f"Buy failed for {pick}: {e}", "WARN")
                else:
                    log(f"Not enough USD for buy; need {USD_PER_TRADE}, have {usd_bal:.2f}", "INFO")
        else:
            log("No candidates (capacity left but universe empty/held).", "INFO")

    # KPI summary line
    positions = load_positions()
    log(f"KPI SUMMARY | dry_run={DRY_RUN} | open={len(positions)} | cap_left={MAX_POSITIONS-len(positions)} | usd≈{usd_bal:.2f}")
    log("Done.")

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        log(f"FATAL: {e}", "ERROR")
        raise
