# trader/broker_robinhood.py
from __future__ import annotations
import inspect
from typing import Optional

try:
    import robin_stocks.robinhood as rh
except Exception as e:
    raise RuntimeError(f"robin_stocks not available: {e}")

class RobinhoodBroker:
    def __init__(
        self,
        username: str,
        password: str,
        totp_secret: Optional[str] = None,
        device_token: Optional[str] = None,
        dry_run: bool = True,
    ):
        self.username = username
        self.password = password
        self.totp_secret = totp_secret
        self.device_token = device_token
        self.dry_run = dry_run

    def login(self) -> None:
        """
        Logs in non-interactively when TOTP is provided; otherwise lets RH handle
        app/SMS approval if possible. Adapts to robin_stocks versions that don't
        accept device_token.
        """
        kwargs = {
            "username": self.username,
            "password": self.password,
            "store_session": False,
        }

        # If we have a TOTP secret, generate the one-time code
        if self.totp_secret:
            try:
                import pyotp
                kwargs["mfa_code"] = pyotp.TOTP(self.totp_secret).now()
            except Exception as e:
                print(f"[Robinhood] Warning: could not generate TOTP: {e}")

        # Add device_token only if this robin_stocks version supports it
        try:
            if "device_token" in inspect.signature(rh.login).parameters and self.device_token:
                kwargs["device_token"] = self.device_token
        except Exception:
            pass

        try:
            rh.login(**kwargs)
        except TypeError:
            # Very old versions: retry without device_token/mfa kwargs
            kwargs.pop("device_token", None)
            kwargs.pop("mfa_code", None)
            rh.login(**kwargs)

    def place_market(self, symbol: str, side: str, notional: float):
        """
        Place a simple fractional notional market order.
        """
        if self.dry_run:
            return {"status": "dry_run", "symbol": symbol, "side": side, "notional": notional}

        if side.lower() == "buy":
            return rh.orders.order_buy_fractional_by_price(symbol, notional, timeInForce="gfd")
        else:
            return rh.orders.order_sell_fractional_by_price(symbol, notional, timeInForce="gfd")

