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

def pick_universe(market):
    if market == "crypto":
        return ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]
    else:
        # Starter safe equities universe
        return ["AAPL", "MSFT", "GOOGL", "AMZN"]

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

if __name__ == "__main__":
    run_trader()
