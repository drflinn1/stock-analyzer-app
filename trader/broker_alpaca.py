# FILE: trader/broker_alpaca.py
# Alpaca Equities broker wrapper — PAPER by default
# - USD-notional market buys (we derive qty from latest price)
# - Position-aware sells
# - Min notional guard
# - DRY_RUN toggle (simulate without hitting the API)
#
# ENV VARS (expected)
#   ALPACA_API_KEY
#   ALPACA_API_SECRET
#   ALPACA_BASE_URL   (default: https://paper-api.alpaca.markets)
#   ALPACA_DATA_URL   (default: https://data.alpaca.markets)
#   MIN_NOTIONAL_USD  (optional; default 1.00)
#   DRY_RUN           ("true"/"false")

from __future__ import annotations
from typing import Optional, Dict, Any
import os, time, math, json
import requests


class AlpacaHTTPError(Exception):
    """Raised when Alpaca HTTP endpoints return a non-2xx status."""
    pass


class AlpacaEquitiesBroker:
    """
    Thin wrapper around Alpaca REST for equities.
    - PAPER by default (base_url defaults to paper)
    - Market buys in USD notional (compute qty from latest price)
    - Position-aware sells
    - Min notional guard
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        data_url: Optional[str] = None,
        dry_run: bool = True,
        min_notional_usd: float = 1.00,
        price_slippage_pad: float = 0.003,  # +0.3% pad when estimating qty
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET", "")
        self.base_url = base_url or os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self.data_url = data_url or os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
        self.dry_run = dry_run
        self.min_notional_usd = float(os.getenv("MIN_NOTIONAL_USD", min_notional_usd))
        self.price_slippage_pad = price_slippage_pad
        self._s = session or requests.Session()

        if not self.dry_run and (not self.api_key or not self.api_secret):
            raise ValueError("ALPACA_API_KEY/SECRET are required for live API calls")

    # ------------------------
    # Internal helpers
    # ------------------------
    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        }

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None, data_api: bool = False) -> Any:
        url = (self.data_url if data_api else self.base_url).rstrip("/") + path
        r = self._s.get(url, headers=self._headers(), params=params, timeout=20)
        if r.status_code >= 300:
            raise AlpacaHTTPError(f"GET {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        url = self.base_url.rstrip("/") + path
        r = self._s.post(url, headers=self._headers(), data=json.dumps(payload), timeout=20)
        if r.status_code >= 300:
            raise AlpacaHTTPError(f"POST {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    # ------------------------
    # Public API
    # ------------------------
    def get_account(self) -> Dict[str, Any]:
        if self.dry_run:
            # Simulated structure
            return {
                "currency": "USD",
                "cash": os.getenv("SIM_CASH", "100000.00"),
                "portfolio_value": os.getenv("SIM_PV", "100000.00"),
                "status": "DRY_RUN",
            }
        return self._get("/v2/account")

    def get_cash(self) -> float:
        acct = self.get_account()
        cash_str = acct.get("cash") or acct.get("buying_power") or "0"
        try:
            return float(cash_str)
        except Exception:
            return 0.0

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        if self.dry_run:
            # No persisted sim positions here; your main.py should persist .state/positions.json
            return {}
        rows = self._get("/v2/positions")
        out: Dict[str, Dict[str, Any]] = {}
        for p in rows:
            out[p["symbol"].upper()] = p
        return out

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        symbol = symbol.upper()
        positions = self.get_positions()
        return positions.get(symbol)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        symbol = symbol.upper()
        try:
            q = self._get(f"/v2/stocks/{symbol}/trades/latest", data_api=True)
            px = q.get("trade", {}).get("p")
            return float(px) if px is not None else None
        except Exception:
            # fallback try quote endpoint
            try:
                q = self._get(f"/v2/stocks/{symbol}/quotes/latest", data_api=True)
                px = q.get("quote", {}).get("ap") or q.get("quote", {}).get("bp")
                return float(px) if px is not None else None
            except Exception:
                return None

    def market_buy_usd(self, symbol: str, usd: float) -> Dict[str, Any]:
        symbol = symbol.upper()
        usd = float(usd)
        if usd < self.min_notional_usd:
            return {"status": "blocked", "reason": f"below_min_notional {usd} < {self.min_notional_usd}"}

        px = self.get_latest_price(symbol)
        if not px or px <= 0:
            return {"status": "blocked", "reason": f"no_price {symbol}"}

        qty = usd / (px * (1 + self.price_slippage_pad))
        qty = max(0.0001, round(qty, 4))  # fractional shares allowed on Alpaca

        if self.dry_run:
            return {
                "status": "simulated",
                "side": "buy",
                "symbol": symbol,
                "notional_usd": round(usd, 2),
                "est_price": round(px, 4),
                "est_qty": qty,
            }

        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
        }
        o = self._post("/v2/orders", payload)
        return {"status": "submitted", "order": o}

    def market_sell_qty(self, symbol: str, qty: float) -> Dict[str, Any]:
        symbol = symbol.upper()
        qty = float(qty)
        if qty <= 0:
            return {"status": "blocked", "reason": "qty<=0"}

        if self.dry_run:
            return {
                "status": "simulated",
                "side": "sell",
                "symbol": symbol,
                "qty": round(qty, 4),
            }

        payload = {
            "symbol": symbol,
            "qty": str(round(qty, 4)),
            "side": "sell",
            "type": "market",
            "time_in_force": "day",
        }
        o = self._post("/v2/orders", payload)
        return {"status": "submitted", "order": o}

    def market_sell_all(self, symbol: str) -> Dict[str, Any]:
        symbol = symbol.upper()
        pos = self.get_position(symbol)
        if not pos:
            return {"status": "blocked", "reason": f"no_position {symbol}"}
        try:
            qty = float(pos.get("qty"))
        except Exception:
            qty = 0.0
        if qty <= 0:
            return {"status": "blocked", "reason": f"zero_qty {symbol}"}
        return self.market_sell_qty(symbol, qty)

    # convenience
    def ping(self) -> bool:
        try:
            _ = self.get_account()
            return True
        except Exception:
            return False


# Optional quick self-test
if __name__ == "__main__":
    b = AlpacaEquitiesBroker(dry_run=True)
    print("Account:", b.get_account())
    print("Cash:", b.get_cash())
    print("AAPL px:", b.get_latest_price("AAPL"))
    print("Buy sim:", b.market_buy_usd("AAPL", 25))
