#!/usr/bin/env python3
"""
kraken_live_trade.py

Thin helper for placing LIVE market orders on Kraken.

Used by kraken_monday_main.py when DRY_RUN == OFF.

Safety:
- Uses environment variables KRAKEN_API_KEY and KRAKEN_API_SECRET.
- Default order type is MARKET (simplest / least surprise).
- Can optionally run in validation mode (no real fills) if
  KRAKEN_VALIDATE_ONLY is set to a truthy value.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from typing import Any, Dict
from urllib.parse import urlencode

import requests

API_BASE = "https://api.kraken.com"


class KrakenLiveError(RuntimeError):
    """Raised when a live Kraken call fails."""


def _get_credentials() -> tuple[str, str]:
    key = os.getenv("KRAKEN_API_KEY") or ""
    secret = os.getenv("KRAKEN_API_SECRET") or ""
    if not key or not secret:
        raise KrakenLiveError("Missing KRAKEN_API_KEY or KRAKEN_API_SECRET")
    return key, secret


def _sign(path: str, data: Dict[str, Any], secret_b64: str) -> str:
    postdata = urlencode(data)
    encoded = (str(data["nonce"]) + postdata).encode()
    message = path.encode() + hashlib.sha256(encoded).digest()
    mac = hmac.new(base64.b64decode(secret_b64), message, hashlib.sha512)
    sigdigest = base64.b64encode(mac.digest()).decode()
    return sigdigest


def _private(path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    key, secret = _get_credentials()
    data = dict(data)  # copy so we don't mutate caller
    data["nonce"] = int(time.time() * 1000)

    headers = {
        "API-Key": key,
        "API-Sign": _sign(path, data, secret),
    }

    resp = requests.post(API_BASE + path, headers=headers, data=data, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    errors = payload.get("error") or []
    if errors:
        # Kraken returns a list of error strings
        raise KrakenLiveError(f"Kraken error(s): {errors}")

    return payload.get("result") or {}


def to_kraken_pair(symbol: str) -> str:
    """
    Convert 'UNI/USD' -> 'UNIUSD', 'BTC/USD' -> 'XBTUSD', etc.

    This is intentionally minimal; most altcoins work as SYMBOLUSD.
    """
    symbol = symbol.upper().strip()

    if symbol == "BTC/USD":
        return "XBTUSD"  # Kraken's classic BTC/USD ticker

    return symbol.replace("/", "")


def place_market_order(symbol: str, side: str, volume: float) -> Dict[str, Any]:
    """
    Place a MARKET order on Kraken.

    symbol: like 'UNI/USD'
    side: 'buy' or 'sell'
    volume: units (e.g. UNI amount)

    Respects KRAKEN_VALIDATE_ONLY: if set to true/1/yes, order is validated
    but NOT actually executed (Kraken 'validate' flag).
    """
    if volume <= 0:
        raise KrakenLiveError(f"Invalid volume {volume}; must be > 0")

    side = side.lower().strip()
    if side not in ("buy", "sell"):
        raise KrakenLiveError(f"Invalid side '{side}', expected 'buy' or 'sell'")

    pair = to_kraken_pair(symbol)

    data: Dict[str, Any] = {
        "pair": pair,
        "type": side,
        "ordertype": "market",
        "volume": f"{volume:.8f}",
    }

    validate_flag = os.getenv("KRAKEN_VALIDATE_ONLY", "false").lower()
    if validate_flag in ("1", "true", "yes", "on"):
        data["validate"] = "true"

    result = _private("/0/private/AddOrder", data)
    return result
