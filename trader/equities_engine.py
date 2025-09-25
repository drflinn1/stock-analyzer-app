"""
Alpaca Equities Engine (Paper or Live; multi-buy + LIVE auto-trim with floor/limits)
- Safe to run every 15m via GitHub Actions
- Skips instantly if market closed
- Buys up to BUY_PER_RUN bracket orders (TP/SL) per run
- Respects MAX_POSITIONS and cash guards
- LIVE-only auto-trim when positions > MAX_POSITIONS:
    * losers first (unrealized_plpc <= TRIM_LOSS_FLOOR)
    * AND in downtrend (Close < SMA20)
    * up to MAX_TRIMS_PER_RUN per run
- Retries transient API errors

ENV:
  ALPACA_API_KEY, ALPACA_API_SECRET
  ALPACA_PAPER ("true"|"false") -> selects Paper or Live endpoint
  DRY_RUN ("true"|"false")
  PER_TRADE_USD, MAX_POSITIONS, BUY_PER_RUN
  TP_PCT, SL_PCT
  TRIM_LOSS_FLOOR  (e.g., -0.01 for -1.0%)  [LIVE only]
  MAX_TRIMS_PER_RUN (e.g., 2)               [LIVE only]
  UNIVERSE (comma list)
  LOG_LEVEL
"""
from __future__ import annotations
import os, sys, math, logging
from dataclasses import dataclass
from typing import List, Tuple
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
ALPACA_API_KEY = (os.getenv("ALPACA_API_KEY") or "").strip()
ALPACA_API_SECRET = (os.getenv("ALPACA_API_SECRET") or "").strip()
ALPACA_PAPER = (os.getenv("ALPACA_PAPER", "true").lower() == "true")  # True=Paper, False=Live

if not (ALPACA_API_KEY and ALPACA_API_SECRET):
    log.error("Missing Alpaca env vars: ALPACA_API_KEY, ALPACA_API_SECRET")
    sys.exit(2)

DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
PER_TRADE_USD = float(os.getenv("PER_TRADE_USD", "25"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))
BUY_PER_RUN = int(os.getenv("BUY_PER_RUN", "1"))
TP_PCT = float(os.getenv("TP_PCT", "0.035"))
SL_PCT = float(os.getenv("SL_PCT", "0.020"))
UNIVERSE = [s.strip().upper() for s in os.getenv("UNIVERSE", "SPY,AAPL").split(",") if s.strip()]

# LIVE trim knobs
TRIM_LOSS_FLOOR = float(os.getenv("TRIM_LOSS_FLOOR", "-0.01"))   # only trim if <= -1.0%
MAX_TRIMS_PER_RUN = int(os.getenv("MAX_TRIMS_PER_RUN", "2"))     # limit trims per run

log.info(f"Alpaca key loaded (ending …{ALPACA_API_KEY[-4:]}); endpoint={'PAPER' if ALPACA_PAPER else 'LIVE'}.")

client = TradingClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_API_SECRET,
    paper=ALPACA_PAPER,
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

def api_err(label: str, e: Exception):
    status = getattr(e, "status_code", None)
    body = getattr(e, "body", None) or getattr(e, "message", None) or str(e)
    log.error(f"{label}: status={status} body={body}")

def yf_downtrend(sym: str) -> bool:
    """True if last close < SMA20 (simple downtrend)."""
    try:
        df = yf.download(sym, period="30d", interval="1d", progress=False)
        if df is None or df.empty:
            return False
        df = df.dropna()
        df["SMA20"] = df["Close"].rolling(20).mean()
        last = df.iloc[-1]
        return not math.isnan(last["SMA20"]) and float(last["Close"]) < float(last["SMA20"])
    except Exception as e:
        log.warning(f"{sym}: downtrend check failed: {e}")
        return False

def qty_to_int(qty_str: str) -> int:
    try:
        return max(1, int(abs(float(qty_str))))
    except Exception:
        return 0

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

# ---------- LIVE auto-trim: losers-first + downtrend + floor + max-per-run ----------
if not ALPACA_PAPER and len(open_positions) > MAX_POSITIONS:
    excess = len(open_positions) - MAX_POSITIONS
    trim_pool: List[Tuple[float, str, int]] = []  # (plpc, symbol, qty_int)

    for p in open_positions:
        sym = p.symbol
        try:
            plpc = float(p.unrealized_plpc or 0.0)  # e.g., -0.023 = -2.3%
        except Exception:
            plpc = 0.0
        if plpc <= TRIM_LOSS_FLOOR and yf_downtrend(sym):
            q = qty_to_int(p.qty)
            if q > 0:
                trim_pool.append((plpc, sym, q))

    if trim_pool:
        trim_pool.sort(key=lambda t: t[0])  # most negative first
        to_close = trim_pool[: min(excess, MAX_TRIMS_PER_RUN)]
        log.info(
            f"LIVE auto-trim: need {excess} -> closing up to {len(to_close)} "
            f"loser(s) in downtrend (floor {TRIM_LOSS_FLOOR:+.2%}) -> {[(s, q) for _, s, q in to_close]}"
        )
        for _, sym, q in to_close:
            if DRY_RUN:
                log.info(f"DRY_RUN: would SELL {sym} x{q} (auto-trim).")
                continue
            try:
                order = submit_order(
                    MarketOrderRequest(
                        symbol=sym,
                        qty=q,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                log.info(f"AUTO-TRIM SELL submitted -> id={order.id} {sym} x{q}")
            except APIError as e:
                api_err(f"AUTO-TRIM SELL failed for {sym}", e)
    else:
        log.info("LIVE auto-trim: over cap but no eligible losers below floor and in downtrend; skipping trim.")

# Recompute positions after possible trim (to set buy slots correctly)
try:
    open_positions = get_positions()
except APIError as e:
    api_err("Post-trim positions fetch failed", e)
    sys.exit(2)
open_symbols = {p.symbol for p in open_positions}

slots_left = max(0, MAX_POSITIONS - len(open_positions))
if slots_left == 0:
    log.info("At MAX_POSITIONS; nothing to buy this run.")
    sys.exit(0)

# ---------- signal: 5/20 SMA cross (autopick) ----------
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
        if math.isnan(last["SMA5"]) or math.isnan(last["SMA20"]):
            continue
        if float(last["SMA5"]) > float(last["SMA20"]):
            candidates.append(Pick(sym, float(last["Close"])))
    except Exception as e:
        log.warning(f"{sym}: data fetch failed: {e}")

if not candidates:
    log.info("No buy signals this run.")
    sys.exit(0)

# Cheaper first to maximize share count with fixed PER_TRADE_USD
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
