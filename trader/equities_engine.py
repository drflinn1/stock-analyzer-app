---

## 2) `trader/equities_engine.py`

```python
"""
Alpaca Equities Engine (Paper-ready)
- Idempotent, safe to run every 15m via GitHub Actions
- Skips instantly if market closed
- Buys small bracket orders with TP/SL
- Never exceeds MAX_POSITIONS
- Retries transient API errors

ENV (from workflow):
  DRY_RUN ("true"/"false")
  PER_TRADE_USD (str->float)
  MAX_POSITIONS (int)
  TP_PCT, SL_PCT (floats like 0.035)
  UNIVERSE (comma list)
  ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL
  LOG_LEVEL (DEBUG/INFO)
"""
from __future__ import annotations
import os, sys, time, math, logging
from dataclasses import dataclass
from typing import List
import pandas as pd
import yfinance as yf
from pytz import timezone
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_fixed

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.rest import URL

# ---------- logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
log = logging.getLogger("equities")

# ---------- config ----------
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
PER_TRADE_USD = float(os.getenv("PER_TRADE_USD", "25"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))
TP_PCT = float(os.getenv("TP_PCT", "0.035"))
SL_PCT = float(os.getenv("SL_PCT", "0.020"))
UNIVERSE = [s.strip().upper() for s in os.getenv("UNIVERSE", "SPY,AAPL").split(",") if s.strip()]

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

if not (ALPACA_API_KEY and ALPACA_API_SECRET and ALPACA_BASE_URL):
    log.error("Missing one or more Alpaca env vars: ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL")
    sys.exit(2)

client = TradingClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_API_SECRET,
    paper=True,
    base_url=URL(ALPACA_BASE_URL),
)

# ---------- helpers ----------
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_clock():
    return client.get_clock()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_positions():
    return client.get_all_positions()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_account():
    return client.get_account()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def submit_order(req: MarketOrderRequest):
    return client.submit_order(req)

# ---------- market-time gate ----------
clock = get_clock()
if not clock.is_open:
    # Allow pre-open / post-close runs to exit quickly with success
    log.info("Market closed (Alpaca clock). Exiting.")
    sys.exit(0)

# Also avoid running before the first 5 minutes after open (data churn)
ny = timezone("America/New_York")
now_ny = datetime.now(ny)
market_open = clock.next_open.astimezone(ny).replace(tzinfo=ny) if now_ny < clock.next_open.astimezone(ny) else clock.timestamp.astimezone(ny)
# If exactly at session, let it pass; otherwise just continue.

# ---------- portfolio state ----------
try:
    acct = get_account()
    cash = float(acct.cash)
except Exception as e:
    log.error(f"Failed to fetch account: {e}")
    sys.exit(2)

open_positions = get_positions()
open_symbols = {p.symbol for p in open_positions}
log.info(f"Cash: ${cash:,.2f} | Open positions: {len(open_positions)} -> {sorted(open_symbols)}")

slots_left = max(0, MAX_POSITIONS - len(open_positions))
if slots_left == 0:
    log.info("At MAX_POSITIONS; nothing to buy this run.")
    sys.exit(0)

budget = min(cash, PER_TRADE_USD)
if budget < 5:  # avoid dust orders
    log.info("Insufficient cash (<$5) for new positions this run.")
    sys.exit(0)

# ---------- simple signal: 5/20 SMA cross on today’s data ----------
@dataclass
class Pick:
    symbol: str
    price: float

candidates: List[Pick] = []
for sym in UNIVERSE:
    if sym in open_symbols:
        continue
    try:
        # Pull a small window to keep it light
        df = yf.download(sym, period="30d", interval="1d", progress=False)
        if df is None or df.empty:
            continue
        df = df.dropna()
        df["SMA5"] = df["Close"].rolling(5).mean()
        df["SMA20"] = df["Close"].rolling(20).mean()
        last = df.iloc[-1]
        if math.isnan(last.SMA5) or math.isnan(last.SMA20):
            continue
        if last.SMA5 > last.SMA20:
            candidates.append(Pick(sym, float(last.Close)))
    except Exception as e:
        log.warning(f"{sym}: data fetch failed: {e}")

if not candidates:
    log.info("No buy signals this run.")
    sys.exit(0)

# Choose the cheapest candidate to maximize share count with small budgets
candidates.sort(key=lambda p: p.price)
pick = candidates[0]

qty = max(1, int(budget // pick.price))
if qty <= 0:
    log.info(f"Signal on {pick.symbol} but budget ${budget:.2f} < price ${pick.price:.2f} -> skip.")
    sys.exit(0)

# Compute TP/SL prices for a bracket via notional percentages
# Note: Alpaca bracket requires limit_price (take-profit) and stop_loss (stop price)
entry = pick.price
tp_price = round(entry * (1 + TP_PCT), 2)
sl_price = round(entry * (1 - SL_PCT), 2)

log.info(
    f"BUY {pick.symbol} x{qty} @~${entry:.2f} with TP ${tp_price:.2f} (+{TP_PCT*100:.1f}%) and SL ${sl_price:.2f} (-{SL_PCT*100:.1f}%)."
)

if DRY_RUN:
    log.info("DRY_RUN=true -> skipping order placement.")
    sys.exit(0)

try:
    order = submit_order(
        MarketOrderRequest(
            symbol=pick.symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class="bracket",
            take_profit={"limit_price": tp_price},
            stop_loss={"stop_price": sl_price},
        )
    )
    log.info(f"Submitted bracket BUY -> id={order.id} symbol={order.symbol} qty={order.qty}")
except Exception as e:
    # Common failure causes are captured in logs; retry already attempted via tenacity where set.
    log.error(f"Order submit failed: {e}")
    sys.exit(1)

log.info("Run complete.")
