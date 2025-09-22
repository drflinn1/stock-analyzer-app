# FILE: trader/broker_alpaca.py
# Alpaca Equities broker wrapper — PAPER by default
# - Minimal deps (uses built-in json, time, math, and requests)
# - Position-aware sells
# - Market buys sized by USD notional (Alpaca submits qty; we derive qty from latest price)
# - Min order guard
# - DRY_RUN toggle (simulates without hitting the API)
#
# ENV VARS (expected)
# ALPACA_API_KEY
# ALPACA_API_SECRET
# ALPACA_BASE_URL (default: https://paper-api.alpaca.markets)
# ALPACA_DATA_URL (default: https://data.alpaca.markets)
# DRY_RUN ("true"/"false")
#
# Typical usage (inside your main.py):
# from trader.broker_alpaca import AlpacaEquitiesBroker
# broker = AlpacaEquitiesBroker(dry_run=os.getenv("DRY_RUN","true").lower()=="true")
# cash = broker.get_cash()
# px = broker.get_latest_price("AAPL")
# broker.market_buy_usd("AAPL", 25)
# broker.market_sell_all("AAPL")


from __future__ import annotations
from typing import Optional, Dict, Any
import os, time, math, json
import requests




class AlpacaHTTPError(Exception):
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
price_slippage_pad: float = 0.003, # +0.3% pad when estimating qty
session: Optional[requests.Session] = None,
) -> None:
self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET", "")
self.base_url = base_url or os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
self.data_url = data_url or os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
self.dry_run = dry_run
self.min_notional_usd = float(os.getenv("MIN_NOTIONAL_USD", min_notional_usd))
self.price_slippage_pad = price_slippage_pad
