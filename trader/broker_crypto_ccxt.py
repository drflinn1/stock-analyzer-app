# trader/broker_crypto_ccxt.py
# CCXT broker wrapper with Kraken-safe USD/ZUSD balance detection.
# Reads either CCXT_* or KRAKEN_* env var names (whichever is present).

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import os, math

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required by broker_crypto_ccxt.py: {e}")

class CCXTCryptoBroker:
    """
    Thin wrapper around CCXT with:
      - Kraken-friendly USD detection (USD or ZUSD)
      - Position-aware sells (placeholder hooks)
      - Min order size guard (placeholder hooks)
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
    ):
        self.exchange_id = exchange_id
        self.dry_run = dry_run

        # Accept either CCXT_* or KRAKEN_* (repo secrets may use either)
        env_api_key = os.getenv("CCXT_API_KEY") or os.getenv("KRAKEN_API_KEY") or ""
        env_api_secret = os.getenv("CCXT_API_SECRET") or os.getenv("KRAKEN_API_SECRET") or ""
        env_api_password = (
            os.getenv("CCXT_API_PASSWORD") or os.getenv("KRAKEN_API_PASSWORD") or None
        )

        self.api_key = api_key if api_key is not None else env_api_key
        self.api_secret = api_secret if api_secret is not None else env_api_secret
        self.api_password = api_password if api_password is not None else env_api_password

        # Build CCXT client
        if not hasattr(ccxt, exchange_id):
            raise SystemExit(f"Unknown exchange id: {exchange_id}")

        klass = getattr(ccxt, exchange_id)
        opts: Dict[str, Any] = {
            "apiKey": self.api_key or "",
            "secret": self.api_secret or "",
            "enableRateLimit": True,
        }
        if self.api_password:
            opts["password"] = self.api_password

        self.ex = klass(opts)
        # Quietly safe defaults
        self.ex.options = getattr(self.ex, "options", {})

        # Debug: show which env source we used (no secrets leaked)
        src = "explicit args" if any([api_key, api_secret, api_password]) else (
            "CCXT_*" if os.getenv("CCXT_API_KEY") else ("KRAKEN_*" if os.getenv("KRAKEN_API_KEY") else "none")
        )
        print(f"2025-09-30 UTC INFO: [broker] CCXT creds source: {src}")

    # -------- helpers -------- #

    def load_markets(self) -> None:
        self.ex.load_markets()

    def fetch_total_balances(self) -> Dict[str, float]:
        bal = self.ex.fetch_balance()
        totals = bal.get("total", {}) or {}
        # Best-effort numeric cast
        out: Dict[str, float] = {}
        for k, v in totals.items():
            try:
                out[k] = float(v)
            except Exception:
                continue
        return out

    def usd_cash(self) -> float:
        totals = self.fetch_total_balances()
        usd = 0.0
        for key in self.USD_KEYS:
            if key in totals:
                try:
                    usd += float(totals[key])
                except Exception:
                    pass
        return usd
