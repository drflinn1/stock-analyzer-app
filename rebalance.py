# rebalance.py
"""
Generates simple BUY/SELL/HOLD signals and a run summary.
Safe to run in CI. Respects DRY_RUN and IGNORE_MARKET_HOURS env vars.

Outputs:
  - rebalance.log  (human-readable log)
  - signals.json   (structured output for later steps)
  - summary.md     (one-line markdown for GitHub job summary)
"""

import os, json, sys, datetime as dt
from typing import List, Dict

# ---------- Config ----------
# Edit this list or load from a file if you prefer.
UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

LOOKBACK = 220         # trading days to download
FAST = 50              # fast SMA window
SLOW = 200             # slow SMA window

# ---------- Env flags ----------
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
IGNORE_MKT = os.getenv("IGNORE_MARKET_HOURS", "false").lower() == "true"

# ---------- Helpers ----------
def now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

def us_market_open_now() -> bool:
    """Very simple gate: Mon–Fri 13:30–20:00 UTC (9:30–16:00 ET).
    Good enough for CI guard; use a market-hours lib if you need holidays."""
    if IGNORE_MKT:
        return True
    now = dt.datetime.now(dt.timezone.utc)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    t = now.time()
    return (dt.time(13, 30) <= t <= dt.time(20, 0))

def write(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def log(msg: str):
    stamp = f"[{now_utc_iso()}] "
    print(stamp + msg)
    with open("rebalance.log", "a", encoding="utf-8") as f:
        f.write(stamp + msg + "\n")

# ---------- Main logic ----------
def compute_signals(tickers: List[str]) -> List[Dict]:
    import pandas as pd
    import yfinance as yf

    out = []
    for t in tickers:
        try:
            df = yf.download(t, period=f"{LOOKBACK}d", interval="1d", auto_adjust=True, progress=False)
            if df.empty or len(df) < SLOW:
                log(f"{t}: insufficient data ({len(df)} bars)")
                continue

            close = df["Close"]
            sma_fast = close.rolling(FAST).mean().iloc[-1]
            sma_slow = close.rolling(SLOW).mean().iloc[-1]
            price = float(close.iloc[-1])

            if pd.isna(sma_fast) or pd.isna(sma_slow):
                log(f"{t}: SMA nan (fast={sma_fast}, slow={sma_slow})")
                continue

            if sma_fast > sma_slow:
                action = "BUY"
            elif sma_fast < sma_slow:
                action = "SELL"
            else:
                action = "HOLD"

            out.append({
                "symbol": t,
                "price": round(price, 4),
                "fast_sma": round(float(sma_fast), 4),
                "slow_sma": round(float(sma_slow), 4),
                "signal": action
            })
        except Exception as e:
            log(f"{t}: error {e!r}")
    return out

def main():
    log(f"Start: dry_run={DRY_RUN}, ignore_market_hours={IGNORE_MKT}")

    if not us_market_open_now():
        log("Market is closed. Exiting early (set IGNORE_MARKET_HOURS=true to override).")
        summary = "Market closed — skipped."
        write("signals.json", json.dumps({
            "generated_at": now_utc_iso(),
            "dry_run": DRY_RUN,
            "ignore_market_hours": IGNORE_MKT,
            "signals": [],
            "summary": summary
        }, indent=2))
        write("summary.md", summary)
        return 0

    signals = compute_signals(UNIVERSE)

    # Optional: turn signals into “orders” here if you add broker integration.
    # For now, just summarize.
    buys = sum(1 for s in signals if s["signal"] == "BUY")
    sells = sum(1 for s in signals if s["signal"] == "SELL")

    summary = f"{len(signals)} signals → BUY: {buys}, SELL: {sells} (DRY_RUN={DRY_RUN})"
    log(summary)

    write("signals.json", json.dumps({
        "generated_at": now_utc_iso(),
        "dry_run": DRY_RUN,
        "ignore_market_hours": IGNORE_MKT,
        "signals": signals,
        "summary": summary
    }, indent=2))
    write("summary.md", summary)

    return 0

if __name__ == "__main__":
    sys.exit(main())
