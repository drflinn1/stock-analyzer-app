import os
import sys
import logging
import ccxt
import yfinance as yf

# ========== Logging Setup ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ========== Env Configs ==========
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
TRADE_AMOUNT = float(os.getenv("TRADE_AMOUNT", "10"))   # spend per buy in QUOTE (e.g., USDT)
DAILY_CAP = float(os.getenv("DAILY_CAP", "50"))
DROP_PCT = float(os.getenv("DROP_PCT", "2.0"))
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT").split(",") if s.strip()]

# Kraken credentials for live trading
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET")

# ========== Broker Setup ==========
def get_exchange():
    opts = {"enableRateLimit": True}
    if not DRY_RUN:
        if not KRAKEN_API_KEY or not KRAKEN_API_SECRET:
            raise RuntimeError("Missing Kraken API credentials in environment.")
        opts.update({"apiKey": KRAKEN_API_KEY, "secret": KRAKEN_API_SECRET})
    return ccxt.kraken(opts)

# ========== Data Fetch ==========
def get_latest_price(symbol: str) -> float:
    base, _ = symbol.split("/")
    t = yf.Ticker(base + "-USD")
    hist = t.history(period="1d", interval="1m")
    if hist.empty:
        raise ValueError(f"No price data for {symbol}")
    return float(hist["Close"].iloc[-1])

# ========== Trading Logic ==========
def should_buy(symbol: str, drop_pct: float) -> bool:
    base, _ = symbol.split("/")
    t = yf.Ticker(base + "-USD")
    hist = t.history(period="5d", interval="1h")
    if len(hist) < 2:
        logger.info(f"{symbol}: not enough data to evaluate gate")
        return False
    last = float(hist["Close"].iloc[-1])
    prev = float(hist["Close"].iloc[-2])
    change_pct = (last - prev) / prev * 100
    logger.info(f"{symbol}: change {change_pct:.2f}% (gate {drop_pct}%)")
    return change_pct <= -drop_pct

def quantize_amount(exchange, symbol: str, amount: float) -> float:
    """Round amount to exchange precision and enforce min size."""
    markets = exchange.load_markets()
    m = markets.get(symbol)
    if not m:
        raise ValueError(f"Symbol not found on exchange: {symbol}")
    q = float(exchange.amount_to_precision(symbol, amount))
    min_amt = (m.get("limits", {}).get("amount", {}) or {}).get("min")
    if min_amt and q < float(min_amt):
        logger.info(f"Adjusted to min amount for {symbol}: {min_amt}")
        q = float(min_amt)
    return q

def place_order(exchange, symbol: str, side: str, quote_spend: float, last_price: float):
    # convert quote spend (e.g., USDT) into base qty (BTC/ETH)
    base_qty = quote_spend / max(last_price, 1e-9)
    if DRY_RUN:
        logger.info(f"[DRY RUN] {side} ~{base_qty:.8f} {symbol} (spend {quote_spend} quote @ {last_price})")
        return
    qty = quantize_amount(exchange, symbol, base_qty)
    logger.info(f"Placing order: {side} {qty} {symbol} (~{quote_spend} quote @ {last_price})")
    order = exchange.create_market_order(symbol, side, qty)
    logger.info(f"Executed order: {order}")

# ========== Alarm Hook ==========
def post_alarm(message: str):
    logger.warning(f"ALARM: {message}")

# ========== Main ==========
def main():
    logger.info("=== START TRADING OUTPUT ===")
    exchange = get_exchange()
    spent_today = 0.0

    for symbol in SYMBOLS:
        try:
            price = get_latest_price(symbol)
            logger.info(f"{symbol} latest price: {price:.2f}")

            if should_buy(symbol, DROP_PCT):
                if spent_today + TRADE_AMOUNT > DAILY_CAP:
                    post_alarm(f"Daily cap {DAILY_CAP} reached, skipping {symbol}")
                    continue
                place_order(exchange, symbol, "buy", TRADE_AMOUNT, price)
                spent_today += TRADE_AMOUNT
            else:
                logger.info(f"No buy signal for {symbol}")
        except Exception as e:
            logger.error(f"Error with {symbol}: {e}")

    logger.info(f"Total spent today: {spent_today:.2f} (cap {DAILY_CAP})")
    logger.info("=== END TRADING OUTPUT ===")

if __name__ == "__main__":
    main()
