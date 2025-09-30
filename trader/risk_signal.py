# trader/risk_signal.py
# Prints a single KPI-style line indicating RISK SIGNAL: ON/OFF
# Heuristics (simple & fast, daily candles via CCXT):
# - Trend: BTC & ETH close > 200SMA AND 50SMA > 200SMA (both)
# - Breadth: >= 3 of {BTC, ETH, SOL, DOGE} have positive 20D return
# - Volatility: ATR(14)/close <= 0.08 on BTC (≤ 8%)
#
# If >=2 of the 3 buckets are true => RISK-ON, else RISK-OFF.

from __future__ import annotations
import math
import os
from typing import List, Tuple
from datetime import datetime, timezone

try:
    import ccxt  # type: ignore
except Exception as e:
    print(f"\x1b[31m[ERROR]\x1b[0m ccxt is required: {e}")
    raise

EXCHANGE_ID = os.getenv("EXCHANGE_ID", "kraken")
UNIVERSE = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]  # pair symbols as used by the exchange

def sma(vals: List[float], n: int) -> float:
    if len(vals) < n:
        return float("nan")
    return sum(vals[-n:]) / n

def atr(ohlcv: List[List[float]], n: int = 14) -> float:
    # ohlcv rows: [ts, open, high, low, close, vol]
    trs: List[float] = []
    for i in range(1, len(ohlcv)):
        _, _, h, l, c = ohlcv[i]
        _, _, _, _, c_prev = ohlcv[i-1]
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        trs.append(tr)
    if len(trs) < n:
        return float("nan")
    return sum(trs[-n:]) / n

def get_exchange():
    cls = getattr(ccxt, EXCHANGE_ID)
    return cls({"enableRateLimit": True})

def fetch_daily(exchange, symbol: str, limit: int = 240):
    # Map to exchange market symbol if needed
    try:
        market = exchange.market(symbol)
        exch_symbol = market["symbol"]
    except Exception:
        exch_symbol = symbol
    return exchange.fetch_ohlcv(exch_symbol, timeframe="1d", limit=limit)

def risk_signal() -> Tuple[str, dict]:
    ex = get_exchange()
    ex.load_markets()
    # 1) Trend (BTC & ETH)
    btc = fetch_daily(ex, "BTC/USD", 240)
    eth = fetch_daily(ex, "ETH/USD", 240)

    def trend_ok(ohlcv: List[List[float]]) -> bool:
        closes = [row[4] for row in ohlcv]
        sma50 = sma(closes, 50)
        sma200 = sma(closes, 200)
        last = closes[-1]
        return (not math.isnan(sma50) and not math.isnan(sma200)
                and last > sma200 and sma50 > sma200)

    t_btc = trend_ok(btc)
    t_eth = trend_ok(eth)
    trend_bucket = (t_btc and t_eth)

    # 2) Breadth: positive 20D return across mini-universe
    breadth_count = 0
    breadth_details = {}
    for sym in UNIVERSE:
        data = fetch_daily(ex, sym, 30)
        closes = [r[4] for r in data]
        if len(closes) >= 21:
            ret20 = (closes[-1] / closes[-21]) - 1.0
            breadth_details[sym] = ret20
            if ret20 > 0:
                breadth_count += 1
        else:
            breadth_details[sym] = float("nan")
    breadth_bucket = breadth_count >= 3  # 3 of 4 positive

    # 3) Volatility: BTC ATR(14) / close <= 8%
    btc_atr = atr(btc, 14)
    btc_close = btc[-1][4]
    vol_ratio = btc_atr / btc_close if btc_close else float("nan")
    vol_bucket = (not math.isnan(vol_ratio)) and (vol_ratio <= 0.08)

    score = sum([trend_bucket, breadth_bucket, vol_bucket])
    signal = "ON" if score >= 2 else "OFF"

    info = {
        "trend": trend_bucket,
        "breadth_positive_count": breadth_count,
        "vol_ok": vol_bucket,
        "vol_ratio": round(vol_ratio, 4) if not math.isnan(vol_ratio) else None,
        "btc_above_200": t_btc,
        "eth_above_200": t_eth,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return signal, info

def main():
    try:
        signal, info = risk_signal()
        # ANSI color: green for ON, red for OFF
        color = "\x1b[32m" if signal == "ON" else "\x1b[31m"
        reset = "\x1b[0m"
        # Single concise KPI line (easy to scan in logs)
        print(
            f"{color}RISK SIGNAL: {signal}{reset} "
            f"(Trend={info['trend']} | Breadth>0: {info['breadth_positive_count']} "
            f"| VolOK={info['vol_ok']} ~ ATR/Close={info['vol_ratio']})"
        )
    except Exception as e:
        print(f"\x1b[31m[ERROR]\x1b[0m risk_signal failed: {e}")
        raise

if __name__ == "__main__":
    main()
