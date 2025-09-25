"""
Alpaca Equities Engine (Paper-ready, multi-buy)
- Safe to run every 15m via GitHub Actions
- Skips instantly if market closed
- Buys up to BUY_PER_RUN bracket orders (TP/SL) per run
- Respects MAX_POSITIONS and cash guards
- Retries transient API errors
"""
from __future__ import annotations
import os, sys, math, logging
from dataclasses import dataclass
from typing import List
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_fixed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError

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
BUY_PER_RUN = int(os.getenv("BUY_PER_RUN", "1"))
TP_PCT = float(os.getenv("TP_PCT", "0.035"))
SL_PCT = float(os.getenv("SL_PCT", "0.020"))
UNIVERSE = [s.strip().upper() for s in os.getenv("UNIVERSE", "SPY,AAPL").split(",") if s.strip()]

ALPACA_API_KEY = (os.getenv("ALPACA_API_KEY") or "").strip()
ALPACA_API_SECRET = (os.getenv("ALPACA_API_SECRET") or "").strip()
if not (ALPACA_API_KEY and ALPACA_API_SECRET):
    log.error("Missing Alpaca env vars: ALPACA_API_KEY, ALPACA_API_SECRET (check GitHub → Settings → Secrets).")
    sys.exit(2)

log.info(f"Alpaca key loaded (ending …{ALPACA_API_KEY[-4:]}); using PAPER endpoint.")
client = TradingClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET, paper=True)

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

def api_err(label: str, e: Exception):
    status = getattr(e, "status_code", None)
    body = getattr(e, "body", None) or getattr(e, "message", None) or str(e)
    log.error(f"{label}: status={status} body={body}")

# ---------- market-time gate ----------
try:
    clock = get_clock()
except APIError as e:
    api_err("Clock fetch failed (likely auth)", e)
    sys.exit(2)
if not clock.is_open:
    log.info("Market closed (Alpaca clock). Exiting.")
    sys.exit(0)

# ---------- account/positions ----------
try:
    acct = get_account()
    cash = float(acct.cash)
except APIError as e:
    api_err("Failed to fetch account", e)
    sys.exit(2)

try:
    open_positions = get_positions()
except APIError as e:
    api_err("Failed to fetch positions", e)
    sys.exit(2)

open_symbols = {p.symbol for p in open_positions}
log.info(f"Cash: ${cash:,.2f} | Open positions: {len(open_positions)} -> {sorted(open_symbols)}")

slots_left = max(0, MAX_POSITIONS - len(open_positions))
if slots_left == 0:
    log.info("At MAX_POSITIONS; nothing to buy this run.")
    sys.exit(0)

# ---------- signal: 5/20 SMA cross ----------
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

# Rank by cheaper first to maximize share count with fixed PER_TRADE_USD
candidates.sort(key=lambda p: p.price)

to_buy = min(slots_left, BUY_PER_RUN, len(candidates))
remaining_cash = cash
buys_done = 0

for pick in candidates[:to_buy]:
    budget = min(remaining_cash, PER_TRADE_USD)
    if budget < 5:
        log.info("Remaining cash this run is < $5 — stopping.")
        break
    qty = max(1, int(budget // pick.price))
    if qty <= 0:
        log.info(f"Skip {pick.symbol}: budget ${budget:.2f} < price ${pick.price:.2f}.")
        continue

    entry = pick.price
    tp_price = round(entry * (1 + TP_PCT), 2)
    sl_price = round(entry * (1 - SL_PCT), 2)

    log.info(f"BUY {pick.symbol} x{qty} @~${entry:.2f} TP ${tp_price:.2f} (+{TP_PCT*100:.1f}%) SL ${sl_price:.2f} (-{SL_PCT*100:.1f}%).")

    if DRY_RUN:
        continue

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
        log.info(f"Submitted BUY -> id={order.id} symbol={order.symbol} qty={order.qty}")
        remaining_cash -= qty * entry
        buys_done += 1
    except APIError as e:
        api_err("Order submit failed", e)

if DRY_RUN:
    log.info("DRY_RUN=true -> no orders placed (log only).")
elif buys_done == 0:
    log.info("No orders placed this run.")
else:
    log.info(f"Run complete: placed {buys_done} order(s).")
