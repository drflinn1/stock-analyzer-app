# trader/broker_robinhood.py
from __future__ import annotations
import time
from typing import Optional

try:
    import robin_stocks.robinhood as rh
except Exception as e:
    raise RuntimeError("robin_stocks is required for RobinhoodBroker") from e


class RobinhoodBroker:
    def __init__(
        self,
        username: str,
        password: str,
        totp_secret: Optional[str] = None,
        dry_run: bool = True,
    ):
        self.username = username
        self.password = password
        self.totp_secret = (totp_secret or "").strip() or None
        self.dry_run = dry_run
        self.logged_in = False

    def _gen_mfa(self) -> Optional[str]:
        if not self.totp_secret:
            return None
        try:
            import pyotp
            return pyotp.TOTP(self.totp_secret).now()
        except Exception:
            return None

    def login(self) -> bool:
        if self.dry_run:
            return True
        mfa_code = self._gen_mfa()
        # NOTE: do NOT pass device_token; some versions of robin_stocks reject it.
        res = rh.login(
            username=self.username,
            password=self.password,
            mfa_code=mfa_code,          # None if you didn't set RH_TOTP_SECRET
            store_session=True,
        )
        # robin_stocks returns a dict on success; None/False on failure
        self.logged_in = bool(res)
        if not self.logged_in:
            raise RuntimeError("Robinhood login failed (MFA/app approval may be required).")
        return True

    def place_market(self, symbol: str, side: str, notional: float):
        """Market order using dollar notional for buys; fractional quantity for sells."""
        if self.dry_run:
            return {"dry_run": True, "symbol": symbol, "side": side, "notional": notional}

        if not self.logged_in:
            self.login()

        side = side.lower().strip()
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")

        # Latest price to translate notional → fractional quantity if needed
        price_str = rh.get_latest_price(symbol)[0]
        price = float(price_str)

        if side == "buy":
            # Fractional dollar notional buy
            return rh.orders.order_buy_fractional_by_price(
                symbol, amountInDollars=float(notional)
            )
        else:
            # Fractional sell: compute an approximate fractional quantity to sell
            qty = round(float(notional) / price, 6)
            return rh.orders.order_sell_fractional_by_quantity(symbol, quantity=qty)
