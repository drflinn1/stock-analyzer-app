#!/usr/bin/env python3
"""
kraken_position_sync.py

Read-only sync between .state/positions.json and live Kraken balances.

Goal (Stage 1 of 'live-read mode'):
- If the bot thinks you hold a position in positions.json but Kraken
  shows ~0 balance for that asset, assume you closed it manually and
  clear positions.json so the bot treats you as flat.
- If Kraken can't be reached or keys are missing, this script is a
  no-op and the bot behaves exactly as before.

This DOES NOT place any orders. It only reads /0/private/Balance.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.parse
from pathlib import Path
from typing import Optional

import requests

STATE_DIR = Path(".state")
POS_PATH = STATE_DIR / "positions.json"

# Map base symbols like "BTC" or "UNI" to Kraken balance asset codes.
ASSET_MAP = {
    "BTC": "XXBT",
    "ETH": "XETH",
    "LTC": "XLTC",
    "XRP": "XXRP",
    "ADA": "ADA",
    "SOL": "SOL",
    "DOGE": "DOGE",
    "LINK": "LINK",
    "UNI": "UNI",
    "AVAX": "AVAX",
    "TRX": "TRX",
    "MATIC": "MATIC",
    "BONK": "BONK",
    "LSK": "LSK",
    "USDT": "USDT",
    "USD": "ZUSD",
}


def kraken_private(path: str, data: dict) -> Optional[dict]:
    """
    Minimal Kraken private API helper calling /0/private/* endpoints.

    Returns the "result" dict on success, or None on error.
    """
    api_key = os.getenv("KRAKEN_API_KEY")
    api_secret = os.getenv("KRAKEN_API_SECRET")

    if not api_key or not api_secret:
        print("[kraken-sync] No KRAKEN_API_KEY/SECRET set; skipping live sync.")
        return None

    url = "https://api.kraken.com" + path
    payload = dict(data)
    payload["nonce"] = str(int(time.time() * 1000))

    postdata = urllib.parse.urlencode(payload)
    encoded = (payload["nonce"] + postdata).encode()

    # Message for HMAC: path + SHA256(nonce + postdata)
    message = path.encode() + hashlib.sha256(encoded).digest()

    secret = base64.b64decode(api_secret)
    mac = hmac.new(secret, message, hashlib.sha512)
    sig = base64.b64encode(mac.digest())

    headers = {
        "API-Key": api_key,
        "API-Sign": sig,
        "User-Agent": "kraken-rotation-sync",
    }

    try:
        resp = requests.post(url, headers=headers, data=postdata, timeout=15)
        resp.raise_for_status()
        j = resp.json()
    except Exception as e:
        print(f"[kraken-sync] HTTP error calling Kraken: {e}")
        return None

    if j.get("error"):
        print(f"[kraken-sync] Kraken API error: {j['error']}")
        return None

    return j.get("result") or {}


def get_balance_for_asset(asset_symbol: str) -> Optional[float]:
    """
    Fetch free balance for a single asset code (e.g., 'UNI', 'XXBT', etc.).
    """
    result = kraken_private("/0/private/Balance", {})
    if result is None:
        return None
    raw = result.get(asset_symbol)
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def main() -> None:
    if not POS_PATH.exists():
        print("[kraken-sync] No .state/positions.json; nothing to sync.")
        return

    try:
        pos = json.loads(POS_PATH.read_text())
    except Exception as e:
        print(f"[kraken-sync] Failed to read positions.json: {e}")
        return

    symbol = pos.get("symbol")
    units = float(pos.get("units") or 0.0)

    if not symbol or units <= 0:
        print("[kraken-sync] positions.json has no active units; nothing to sync.")
        return

    base = symbol.split("/")[0]
    asset_code = ASSET_MAP.get(base, base)

    live_units = get_balance_for_asset(asset_code)
    if live_units is None:
        print("[kraken-sync] Could not fetch live balance; leaving state unchanged.")
        return

    # If Kraken balance is effectively zero compared to state, clear local position.
    # Two guards:
    #   - absolute very small
    #   - and < 1% of what state thinks you own
    if (live_units < 1e-8) or (units > 0 and live_units < units * 0.01):
        print(
            f"[kraken-sync] Kraken shows ~0 {base} (live {live_units}, "
            f"state {units}). Assuming position closed manually; "
            "clearing .state/positions.json."
        )
        try:
            POS_PATH.unlink()
        except FileNotFoundError:
            pass
        return

    print(
        f"[kraken-sync] Live Kraken balance for {base}: {live_units} "
        f"(state units: {units}). Leaving state as-is."
    )


if __name__ == "__main__":
    main()
