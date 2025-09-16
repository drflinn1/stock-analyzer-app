#!/usr/bin/env python3
"""
Tax Ledger & Reserve helper
- Append every SELL to data/tax_ledger.csv with realized P/L and tax reserve
- Simple, file-based; safe defaults; no external deps
"""

import csv, os, datetime as dt
from typing import Optional

DEFAULT_LEDGER_PATH = os.getenv("TAX_LEDGER_PATH", "data/tax_ledger.csv")
DEFAULT_RESERVE_RATE = float(os.getenv("TAX_RESERVE_RATE", "0.30"))  # 30% default
DEFAULT_STATE_RATE   = float(os.getenv("STATE_TAX_RATE", "0.00"))    # add if desired

CSV_HEADERS = [
    "timestamp_iso", "year", "market", "symbol",
    "side", "qty", "avg_price_usd",
    "proceeds_usd", "cost_basis_usd", "fees_usd",
    "profit_usd", "holding_period_days",
    "short_or_long", "reserve_rate", "state_rate", "reserved_usd",
    "run_id", "trade_id"
]

def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def _holding_term(days: Optional[float]) -> str:
    if days is None: return ""
    try:
        return "LONG" if float(days) >= 365.0 else "SHORT"
    except Exception:
        return ""

class TaxLedger:
    def __init__(
        self,
        ledger_path: str = DEFAULT_LEDGER_PATH,
        reserve_rate: float = DEFAULT_RESERVE_RATE,
        state_rate: float = DEFAULT_STATE_RATE,
    ):
        self.ledger_path = ledger_path
        self.reserve_rate = reserve_rate
        self.state_rate = state_rate
        _ensure_parent_dir(self.ledger_path)
        self._ensure_headers()

    def _ensure_headers(self):
        if not os.path.exists(self.ledger_path) or os.path.getsize(self.ledger_path) == 0:
            with open(self.ledger_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(CSV_HEADERS)

    def record_sell(
        self,
        *,
        market: str,            # "crypto" | "equities"
        symbol: str,            # e.g., "BTC/USD" or "AAPL"
        qty: float,
        avg_price_usd: float,   # average fill price of the SELL
        proceeds_usd: float,    # cash in from the sell (gross)
        cost_basis_usd: float,  # cost of the units you just sold
        fees_usd: float = 0.0,
        holding_period_days: Optional[float] = None,
        run_id: Optional[str] = None,   # e.g., GitHub run id/sha
        trade_id: Optional[str] = None  # your internal id
    ) -> dict:
        """
        Call this AFTER a SELL fills and you know realized P/L on that closed lot.
        Returns a dict with reserved_usd and profit_usd for on-screen logging.
        """
        ts = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        year = ts[:4]
        profit_usd = round((proceeds_usd - fees_usd) - cost_basis_usd, 2)

        # Reserve only on profits (never on losses)
        combined_rate = max(0.0, self.reserve_rate) + max(0.0, self.state_rate)
        reserved_usd = round(max(0.0, profit_usd) * combined_rate, 2)

        row = [
            ts, year, market.lower(), symbol,
            "SELL", f"{qty:.8f}", f"{avg_price_usd:.8f}",
            f"{proceeds_usd:.2f}", f"{cost_basis_usd:.2f}", f"{fees_usd:.2f}",
            f"{profit_usd:.2f}", f"{holding_period_days:.2f}" if holding_period_days is not None else "",
            _holding_term(holding_period_days), f"{self.reserve_rate:.4f}", f"{self.state_rate:.4f}", f"{reserved_usd:.2f}",
            run_id or "", trade_id or ""
        ]

        with open(self.ledger_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

        return {
            "timestamp_iso": ts,
            "profit_usd": profit_usd,
            "reserved_usd": reserved_usd,
            "reserve_rate": self.reserve_rate,
            "state_rate": self.state_rate,
            "ledger_path": self.ledger_path
        }

    def reserve_balance(self) -> float:
        """Sum of all reserved_usd in the ledger (useful to compare with your savings bucket)."""
        total = 0.0
        if not os.path.exists(self.ledger_path):
            return 0.0
        with open(self.ledger_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    total += float(r.get("reserved_usd", "0") or 0)
                except Exception:
                    pass
        return round(total, 2)
