import os
import sys
import time
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
TRADE_AMOUNT = float(os.getenv("TRADE_AMOUNT", "10"))
DAILY_CAP = float(os.getenv("DAILY_CAP", "50"))
DROP_PCT = float(os.getenv("DROP_PCT", "2.0"))  # buy trigger % drop
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT").split(",")

# ========== Broker Setup ==========
def get_exchange():
    exchange = ccxt.kraken({
        "enableRateLimit": True,
    })
    return exchange

# ========== Data Fetch ==========
def get_latest_price(symbol: str):
    base, quote = symbol.split("/")
    ticker = yf.Ticker(base + "-USD")
    hist = ticker.history(period="1d", interval="1m")
    if hist.empty:
        raise ValueError(f"No price data for {symbol}")
    return hist["Close"].iloc[-1]

# ========== Trading Logic ==========
def should_buy(symbol: str, drop_pct: float) -> bool:
    base, quote = symbol.split("/")
    ticker = yf.Ticker(base + "-USD")
    hist = ticker.history(period="5d", interval="1h")
    if len(hist) < 2:
        return False
    last = hist["Close"].iloc[-1]
    prev = hist["Close"].iloc[-2]
    change_pct = (last - prev) / prev * 100
    logger.info(f"{symbol}: change {change_pct:.2f}% (gate {drop_pct}%)")
    return change_pct <= -drop_pct

def place_order(exchange, symbol: str, side: str, amount: float):
    if DRY_RUN:
        logger.info(f"[DRY RUN] {side} {amount} {symbol}")
        return
    try:
        order = exchange.create_market_order(symbol, side, amount)
        logger.info(f"Executed order: {order}")
    except Exception as e:
        logger.error(f"Order failed for {symbol}: {e}")

# ========== Alarm Hook ==========
def post_alarm(message: str):
    # Placeholder for Slack/email later
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
                place_order(exchange, symbol, "buy", TRADE_AMOUNT)
                spent_today += TRADE_AMOUNT
            else:
                logger.info(f"No buy signal for {symbol}")

        except Exception as e:
            logger.error(f"Error with {symbol}: {e}")

    logger.info(f"Total spent today: {spent_today:.2f} (cap {DAILY_CAP})")
    logger.info("=== END TRADING OUTPUT ===")

if __name__ == "__main__":
    main()
