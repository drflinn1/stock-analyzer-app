from dataclasses import dataclass
from typing import Optional

try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None

def norm_symbol(sym: str) -> str:
    # 'BTC-USD' -> 'BTC/USD' for ccxt
    return sym.replace("-", "/").upper()

@dataclass
class CCXTCryptoBroker:
    exchange_id: str            # e.g., "kraken", "binanceus", "coinbasepro"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_password: Optional[str] = None  # some exchanges need a passphrase
    dry_run: bool = True

    def _client(self):
        if ccxt is None:
            raise RuntimeError("ccxt not installed")
        klass = getattr(ccxt, self.exchange_id)
        return klass({
            "apiKey": self.api_key or "",
            "secret": self.api_secret or "",
            "password": self.api_password or None,
            "enableRateLimit": True,
        })

    def place_market_notional(self, symbol: str, side: str, notional_usd: float):
        if self.dry_run:
            return {"dry_run": True, "broker": self.exchange_id, "symbol": symbol,
                    "side": side, "notional_usd": notional_usd}
        ex = self._client()
        market = norm_symbol(symbol)
        price = float(ex.fetch_ticker(market)["last"])
        amount = float(notional_usd) / price
        return ex.create_order(symbol=market, type="market", side=side, amount=amount)
