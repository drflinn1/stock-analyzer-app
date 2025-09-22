import os
import logging

try:
    import robin_stocks.robinhood as r
except ImportError:
    raise ImportError("Please install robin_stocks: pip install robin_stocks")

class RobinhoodBroker:
    def __init__(self):
        self.username = os.getenv("ROBINHOOD_USERNAME")
        self.password = os.getenv("ROBINHOOD_PASSWORD")
        self.mfa = os.getenv("ROBINHOOD_MFA")
        self.logged_in = False

    def login(self):
        if not self.logged_in:
            try:
                r.login(self.username, self.password, mfa_code=self.mfa)
                self.logged_in = True
                logging.info("Logged in to Robinhood.")
            except Exception as e:
                logging.error(f"Robinhood login failed: {e}")
                raise

    def get_balance(self):
        self.login()
        try:
            profile = r.profiles.load_account_profile()
            return float(profile["cash"])
        except Exception as e:
            logging.error(f"Failed to fetch Robinhood balance: {e}")
            return 0.0

    def buy(self, symbol: str, usd_amount: float):
        self.login()
        try:
            quote = float(r.stocks.get_latest_price(symbol)[0])
            qty = round(usd_amount / quote, 4)
            if os.getenv("DRY_RUN", "true").lower() == "true":
                logging.info(f"[DRY RUN] Buy {qty} {symbol} at {quote}")
                return True
            r.orders.order_buy_fractional_by_price(symbol, usd_amount, timeInForce="gfd")
            logging.info(f"Placed buy: {symbol} for ${usd_amount}")
            return True
        except Exception as e:
            logging.error(f"Buy failed: {e}")
            return False

    def sell(self, symbol: str, qty: float):
        self.login()
        try:
            if os.getenv("DRY_RUN", "true").lower() == "true":
                logging.info(f"[DRY RUN] Sell {qty} {symbol}")
                return True
            r.orders.order_sell_fractional_by_quantity(symbol, qty, timeInForce="gfd")
            logging.info(f"Placed sell: {qty} {symbol}")
            return True
        except Exception as e:
            logging.error(f"Sell failed: {e}")
            return False
