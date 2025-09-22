# CCXT broker wrapper with Kraken-safe USD/ZUSD balance detection.

from typing import Optional, Dict, Any, Tuple
import math
import os
import logging
import ccxt


class CCXTCryptoBroker:
    """
    Thin wrapper around CCXT with:
      - Kraken-friendly USD detection (USD or ZUSD)
      - Position-aware sells
      - Min order size guard
      - Market buys sized by USD notional
    """

    USD_KEYS = ("USD", "ZUSD")  # Kraken may report ZUSD

    def __init__(
        self,
        exchange_id: str = "kraken",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        api_password: Optional[str] = None,  # for exchanges that need it
        dry_run: bool = True,
        enable_private: bool = True,
    ):
        self.dry_run = dry_run

        if not exchange_id:
            raise ValueError("exchange_id required (e.g., 'kraken')")

        # Map common aliases
        aliases = {"krakenpro": "kraken"}
        exchange_id = aliases.get(exchange_id.lower(), exchange_id.lower())

        # Pull from env if not provided
        if api_key is None:
            api_key = os.getenv("KRAKEN_API_KEY", "")
        if api_secret is None:
            api_secret = os.getenv("KRAKEN_API_SECRET", "")

        if not enable_private:
            api_key, api_secret, api_password = "", "", None

        klass = getattr(ccxt, exchange_id)
        self.exchange = klass(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "password": api_password,
                "enableRateLimit": True,
            }
        )

        # Lazy-load markets; many helpers rely on them
        self._markets_loaded = False

    # ---------- Market / price helpers ---------- #

    def _ensure_markets(self) -> None:
        if not self._markets_loaded or not getattr(self.exchange, "markets", None):
            self.exchange.load_markets()
            self._markets_loaded = True

    def _round_amount(self, symbol: str, amount: float) -> float:
        self._ensure_markets()
        m = self.exchange.market(symbol)
        precision = (m.get("precision") or {}).get("amount")
        if precision is not None:
            step = 10 ** (-precision)
            amount = math.floor(amount / step) * step

        limits = m.get("limits") or {}
        min_amt = (limits.get("amount") or {}).get("min")
        if min_amt and amount < float(min_amt):
            return 0.0
        return float(amount)

    def fetch_price(self, symbol: str) -> float:
        t = self.exchange.fetch_ticker(symbol)
        last = t.get("last")
        if last:
            return float(last)
        bid = t.get("bid") or 0
        ask = t.get("ask") or 0
        if bid and ask:
            return (float(bid) + float(ask)) / 2.0
        if t.get("close"):
            return float(t["close"])
        raise ValueError(f"Could not determine price for {symbol}: {t}")

    # ---------- Balance helpers (USD/ZUSD) ---------- #

    def _fetch_balance(self) -> Dict[str, Any]:
        return self.exchange.fetch_balance()

    def get_free_cash(
        self, prefer: Tuple[str, ...] = ("USD", "ZUSD", "USDT")
    ) -> Tuple[str, float]:
        bal = self._fetch_balance()
        free = bal.get("free") or {}

        # First, return the first positive balance from the preferred list
        for key in prefer:
            if key in free and free[key] is not None:
                amt = float(free[key]) or 0.0
                if amt > 0:
                    return key, amt

        # If none positive, return the first present (may be 0.0)
        for key in prefer:
            if key in free:
                return key, float(free.get(key) or 0.0)

        return "", 0.0

    def usd_free(self) -> float:
        """Convenience: free USD on Kraken (handles USD or ZUSD)."""
        _, amt = self.get_free_cash(prefer=self.USD_KEYS)
        return amt

    # ---------- Trading helpers ---------- #

    def place_market_notional(
        self, symbol: str, usd_amount: float, fee_buffer: float = 0.001
    ) -> Dict[str, Any]:
        """
        Market buy sized by USD notional. Converts to base amount using live price.
        A small fee_buffer keeps cost <= usd_amount.
        """
        if usd_amount <= 0:
            raise ValueError("usd_amount must be > 0")

        price = self.fetch_price(symbol)
        base_amt = (usd_amount / price) * (1.0 - fee_buffer)
        base_amt = self._round_amount(symbol, base_amt)
        if base_amt <= 0:
            raise ValueError(
                f"Computed amount too small for {symbol} with usd_amount={usd_amount}"
            )

        if self.dry_run:
            return {
                "id": "dryrun",
                "type": "market",
                "side": "buy",
                "symbol": symbol,
                "amount": base_amt,
                "price_used": price,
                "status": "dry_run_only",
            }

        return self.exchange.create_order(
            symbol, type="market", side="buy", amount=base_amt
        )

    def market_sell_all(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Sell full available base position of `symbol` (e.g., BTC for BTC/USD)."""
        bal = self._fetch_balance()
        base = symbol.split("/")[0]
        free_base = float((bal.get("free") or {}).get(base, 0.0) or 0.0)
        free_base = self._round_amount(symbol, free_base)
        if free_base <= 0:
            return None

        if self.dry_run:
            return {
                "id": "dryrun",
                "type": "market",
                "side": "sell",
                "symbol": symbol,
                "amount": free_base,
                "status": "dry_run_only",
            }

        return self.exchange.create_order(
            symbol, type="market", side="sell", amount=free_base
        )


# ===== Standard interface adapter for main.py =====
# Provides: get_balance(), buy(symbol, usd_amount), sell(symbol, qty)
# Inherits your existing CCXTCryptoBroker and assumes `self.exchange` is a ccxt instance.

DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

def _usd_from_ccxt_balance(bal: dict) -> float:
    # direct USD
    for k in ("USD", "ZUSD"):
        v = bal.get(k)
        if isinstance(v, dict):
            for sub in ("free", "total", "available"):
                if isinstance(v.get(sub), (int, float)):
                    return float(v[sub])
        if isinstance(v, (int, float)):
            return float(v)
    # stables (assume 1:1)
    for k in ("USDT", "USDC"):
        v = bal.get(k)
        if isinstance(v, dict) and isinstance(v.get("free"), (int, float)):
            return float(v["free"])
    free = bal.get("free")
    if isinstance(free, dict):
        for k in ("USD", "ZUSD", "USDT", "USDC"):
            if isinstance(free.get(k), (int, float)):
                return float(free[k])
    return 0.0

class CryptoBroker(CCXTCryptoBroker):
    """
    Minimal shim so main.py can call a consistent API.
    Your original CCXTCryptoBroker stays untouched above.
    """
    def _ex(self):
        ex = getattr(self, "exchange", None)
        if ex is None:
            raise RuntimeError("CCXTCryptoBroker has no `exchange` (ccxt) set")
        return ex

    def get_balance(self) -> float:
        try:
            bal = self._ex().fetch_balance()
            usd = _usd_from_ccxt_balance(bal)
            logging.info(f"[ccxt] USD balance detected: ${usd:.2f}")
            return float(usd)
        except Exception as e:
            logging.error(f"[ccxt] fetch_balance failed: {e}")
            return 0.0

    def _last_price(self, symbol: str) -> float:
        ex = self._ex()
        try:
            t = ex.fetch_ticker(symbol)
            if t and t.get("last"):
                return float(t["last"])
        except Exception:
            pass
        ob = ex.fetch_order_book(symbol)
        if ob.get("bids"):
            return float(ob["bids"][0][0])
        if ob.get("asks"):
            return float(ob["asks"][0][0])
        raise RuntimeError(f"No price for {symbol}")

    def buy(self, symbol: str, usd_amount: float) -> bool:
        try:
            price = self._last_price(symbol)
            amount = float(usd_amount) / max(price, 1e-12)
            ex = self._ex()
            if hasattr(ex, "amount_to_precision"):
                amount = float(ex.amount_to_precision(symbol, amount))
            if DRY_RUN:
                logging.info(f"[DRY RUN] ccxt BUY {symbol} amount={amount} (~${usd_amount:.2f}) @~{price}")
                return True
            ex.create_market_buy_order(symbol, amount)
            logging.info(f"[LIVE] ccxt BUY {symbol} amount={amount}")
            return True
        except Exception as e:
            logging.error(f"[ccxt] buy failed: {e}")
            return False

    def sell(self, symbol: str, qty: float) -> bool:
        try:
            ex = self._ex()
            q = float(qty)
            if hasattr(ex, "amount_to_precision"):
                q = float(ex.amount_to_precision(symbol, q))
            if DRY_RUN:
                logging.info(f"[DRY RUN] ccxt SELL {symbol} qty={q}")
                return True
            ex.create_market_sell_order(symbol, q)
            logging.info(f"[LIVE] ccxt SELL {symbol} qty={q}")
            return True
        except Exception as e:
            logging.error(f"[ccxt] sell failed: {e}")
            return False
# ===== end adapter =====
