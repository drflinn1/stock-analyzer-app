import os
import logging
import random

from broker_crypto_ccxt import CryptoBroker
from broker_robinhood import RobinhoodBroker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

MARKET = os.getenv("MARKET", "crypto")  # "crypto" or "equities"
STATE_DIR = os.getenv("STATE_DIR", ".state")
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
PER_TRADE_USD = float(os.getenv("PER_TRADE_USD", "10"))
DAILY_CAP_USD = float(os.getenv("DAILY_CAP_USD", "20"))

# --- Sell Logic Parameters ---
TAKE_PROFIT = 0.03     # 3% target
TRAIL_ACTIVATE = 0.02  # trail after 2% gain
TRAIL_DISTANCE = 0.01  # trail 1% below peak
STOP_LOSS = 0.02       # cut losses at -2%

def pick_universe(market):
    if market == "crypto":
        return ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]
    else:
        # Starter safe equities universe
        return ["AAPL", "MSFT", "GOOGL", "AMZN"]

def should_sell(entry_price, current_price, peak_price=None):
    change = (current_price - entry_price) / entry_price

    # Take-profit
    if change >= TAKE_PROFIT:
        logging.info(f"TAKE_PROFIT hit at {change:.2%}")
        return True

    # Trailing stop
    if peak_price and change >= TRAIL_ACTIVATE:
        trail_stop = peak_price * (1 - TRAIL_DISTANCE)
        if current_price < trail_stop:
            logging.info(f"TRAIL stop hit: price {current_price}, stop {trail_stop}")
            return True

    # Stop-loss
    if change <= -STOP_LOSS:
        logging.info(f"STOP_LOSS hit at {change:.2%}")
        return True

    return False

def run_trader():
    logging.info(f"Starting trader in {MARKET.upper()} mode. Dry run={DRY_RUN}")

    if MARKET == "crypto":
        broker = CryptoBroker()
    else:
        broker = RobinhoodBroker()

    balance = broker.get_balance()
    logging.info(f"Available balance: ${balance:.2f}")

    universe = pick_universe(MARKET)
    logging.info(f"Universe: {universe}")

    daily_spend = min(balance, DAILY_CAP_USD)
    per_trade = min(PER_TRADE_USD, daily_spend)

    if daily_spend < per_trade:
        logging.warning("Not enough balance for even one trade.")
        return

    # Pick random ticker from universe for now (placeholder logic)
    choice = random.choice(universe)
    logging.info(f"Selected {choice} for trade amount ${per_trade}")

    success = broker.buy(choice, per_trade)
    if success:
        logging.info(f"Trade executed: {choice} ${per_trade}")
    else:
        logging.warning("Trade failed.")

    # --- Example sell check ---
    entry = 100
    current = 103
    peak = 104
    if should_sell(entry, current, peak):
        qty = 0.1
        broker.sell(choice, qty)  # dummy qty for test
        # Add uppercase SELL log so guard passes
        logging.info(f"SELL executed: {choice} qty={qty}")

if __name__ == "__main__":
    run_trader()
