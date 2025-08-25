# trader/broker_crypto_ccxt.py
import ccxt
from typing import Optional, Dict, Any

class CCXTCryptoBroker:
    """
    Thin wrapper around CCXT with a 'place_market_notional' helper and
    built-in guards:
      - position-aware sells (skip if no/insufficient base asset)
      - min order size check (skip if amount < exchange min)
    """

    def __init__(
        self,
        exchange_id: str = "kraken",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        api_password: Optional[str] = None,
        dry_run: bool = True,
    ):
        self.dry_run = dry_run
        if not exchange_id:
            raise ValueError("exchange_id required (e.g., 'kraken')")

        klass = getattr(ccxt, exchange_id)
        self.exchange = klass({
            "apiKey": api_key or "",
            "secret": api_secret or "",
            # Kraken ignores 'password' for spot, but include for xchgs that need it
            "password": api_password or "",
            "enableRateLimit": True,
            # safer post-only default off for market orders
            "options": {"defaultType": "spot"},
        })
        # Load markets so we can read precision/limits/base asset, etc.
        self.exchange.load_markets()

    # ---------- helpers ----------

    @staticmethod
    def _norm_symbol(symbol: str) -> str:
        # Accept "BTC-USD" or "BTC/USD"; normalize for ccxt
        return symbol.replace("-", "/").upper()

    def _market(self, symbol: str) -> Dict[str, Any]:
        s = self._norm_symbol(symbol)
        return self.exchange.market(s)

    def _mid_price(self, symbol: str) -> Optional[float]:
        s = self._norm_symbol(symbol)
        # Try best of both: order book mid, then ticker last
        try:
            ob = self.exchange.fetch_order_book(s, limit=5)
            bid = ob["bids"][0][0] if ob["bids"] else None
            ask = ob["asks"][0][0] if ob["asks"] else None
            if bid and ask:
                return (bid + ask) / 2.0
        except Exception:
            pass
        try:
            t = self.exchange.fetch_ticker(s)
            return float(t.get("last") or t.get("close") or 0) or None
        except Exception:
            return None

    def _free_base(self, symbol: str) -> float:
        """Return free amount of the base asset for a symbol (e.g., BTC for BTC/USD)."""
        s = self._norm_symbol(symbol)
        m = self.exchange.market(s)
        base = m.get("base", "")
        # Kraken historically uses XBT; ccxt normalizes to BTC, but we'll check both.
        aliases = [base]
        if base == "BTC":
            aliases.append("XBT")
        try:
            bal = self.exchange.fetch_balance()
            free = 0.0
            # ccxt typically exposes balances under bal['free'][ASSET]
            free_map = (bal.get("free") or {})
            for a in aliases:
                if a in free_map:
                    free = max(free, float(free_map.get(a) or 0.0))
            # Some exchanges also keep a top-level key
            for a in aliases:
                if a in bal:
                    free = max(free, float((bal.get(a) or {}).get("free") or 0.0))
            return free
        except Exception:
            return 0.0

    # ---------- public ----------

    def place_market_notional(self, symbol: str, side: str, notional_usd: float) -> Dict[str, Any]:
        """
        Place a market order sized by *notional_usd*.
        For SELL: clamps to available base; skips if no position.
        Returns either the ccxt order dict or a small dict describing a skip/dry run.
        """
        s = self._norm_symbol(symbol)
        side = side.lower().strip()
        if side not in ("buy", "sell"):
            return {"error": f"Invalid side '{side}'", "symbol": symbol}

        price = self._mid_price(s)
        if not price or price <= 0:
            return {"error": "Price unavailable", "symbol": symbol}

        # Convert notional (USD) to amount of base to buy/sell
        mkt = self._market(s)
        raw_amt = (float(notional_usd) / float(price))
        # Respect precision
        try:
            amount = float(self.exchange.amount_to_precision(s, raw_amt))
        except Exception:
            amount = raw_amt

        # SELL guard: don’t try to sell what we don’t have
        if side == "sell":
            free_base = self._free_base(s)
            # Clamp to 98% of free to leave dust for fees/rounding
            max_sell = max(0.0, free_base * 0.98)
            if max_sell <= 0:
                return {
                    "skipped": "no_position",
                    "reason": "base_free=0",
                    "symbol": symbol,
                    "side": side,
                    "notional_usd": notional_usd,
                }
            if amount > max_sell:
                amount = max_sell

        # Exchange min amount check
        min_amt = 0.0
        try:
            min_amt = float(((mkt.get("limits") or {}).get("amount") or {}).get("min") or 0.0)
        except Exception:
            pass
        if min_amt and amount < min_amt:
            return {
                "skipped": "amount_below_min",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "min_amount": min_amt,
                "notional_usd": notional_usd,
            }

        if self.dry_run:
            return {
                "dry_run": True,
                "broker": self.exchange.id,
                "symbol": symbol,
                "side": side,
                "notional_usd": notional_usd,
                "amount_est": amount,
            }

        try:
            # Market orders: provide amount (base currency units)
            order = self.exchange.create_order(s, "market", side, amount)
            return order
        except ccxt.BaseError as e:
            # Bubble a structured error so the caller prints neatly
            return {"error": str(e), "symbol": symbol, "side": side, "amount": amount, "notional_usd": notional_usd}
