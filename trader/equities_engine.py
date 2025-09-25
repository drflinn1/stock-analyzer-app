"""
Alpaca Equities Engine (Paper-ready)
- Idempotent, safe to run every 15m via GitHub Actions
- Skips instantly if market closed
- Buys small bracket orders with TP/SL
- Never exceeds MAX_POSITIONS
- Retries transient API errors
"""
from __future__ import annotations
import os, sys, math, logging
from dataclasses import dataclass
from typing import List
import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_fixed

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

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
# If you later need custom endpoints, we can switch SDKs or add an env switch.
# For alpaca-py TradingClient, `paper=True` selects the paper base URL automatically.

if not (ALPACA_API_KEY and ALPACA_API_SECRET):
    log.error("Missing Alpaca env vars: ALPACA_API_KEY, ALPACA_API_SECRET")
    sys.exit(2)

client = TradingClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_API_SECRET,
    paper=True,           # selects https://paper-api.alpaca.markets internally
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
    log.info("Market closed (Alpaca clock). Exiting.")
    sys.exit(0)

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
if budget < 5:
    log.info("Insufficient cash (<$5) for new positions this run.")
    sys.exit(0)

# ---------- simple signal: 5/20 SMA cross ----------
@dataclass
class Pick:
    symbol: str
    price: float

candidates: List[Pick] = []
for sym in UNIVERSE:
    if sym in open_symbols:
        continue
    try:
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

candidates.sort(key=lambda p: p.price)
pick = candidates[0]

qty = max(1, int(budget // pick.price))
if qty <= 0:
    log.info(f"Signal on {pick.symbol} but budget ${budget:.2f} < price ${pick.price:.2f} -> skip.")
    sys.exit(0)

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
    log.error(f"Order submit failed: {e}")
    sys.exit(1)

log.info("Run complete.")
