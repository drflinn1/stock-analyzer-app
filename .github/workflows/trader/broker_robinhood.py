from dataclasses import dataclass
from typing import Optional
import pyotp

try:
    import robin_stocks.robinhood as rh
except Exception:
    rh = None

@dataclass
class RobinhoodBroker:
    username: str
    password: str
    totp_secret: Optional[str] = None
    device_token: Optional[str] = None
    dry_run: bool = True

    def login(self):
        if rh is None:
            raise RuntimeError("robin_stocks not installed")
        mfa = pyotp.TOTP(self.totp_secret).now() if self.totp_secret else None
        rh.login(
            username=self.username,
            password=self.password,
            mfa_code=mfa,
            store_session=False,
            expiresIn=24*60*60,
            device_token=self.device_token,
        )

    def place_market(self, symbol: str, side: str, quantity: float = None, notional: float = None):
        if self.dry_run:
            return {"dry_run": True, "broker": "robinhood", "symbol": symbol,
                    "side": side, "qty": quantity, "notional": notional}
        if rh is None:
            raise RuntimeError("robin_stocks not installed")
        if side.lower() == "buy":
            if notional is not None:
                return rh.orders.order_buy_fractional_by_price(symbol, amountInDollars=float(notional))
            elif quantity is not None:
                return rh.orders.order_buy_market(symbol, quantity=quantity)
        else:
            if quantity is not None:
                return rh.orders.order_sell_market(symbol, quantity=quantity)
            # fallback fractional sell if qty not provided
            return rh.orders.order_sell_fractional_by_quantity(symbol, quantity=0.0001)
