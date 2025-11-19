#!/usr/bin/env python3
"""
kraken_live_state.py

Observation-only helper for the LIVE 1-coin rotation bot.

Goal:
- Mirror a tiny bit of state into `.state/positions.json`
- Emit a human-readable `.state/run_summary.md`
- Never send any trading orders.

If anything goes wrong (e.g., API error), it still writes a summary
explaining what happened so the GitHub Actions artifact upload has
something useful.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import requests

API_BASE = "https://api.kraken.com"
STATE_DIR = Path(".state")
POSITIONS_PATH = STATE_DIR / "positions.json"
SUMMARY_PATH = STATE_DIR / "run_summary.md"


@dataclass
class LiveConfig:
    mode: str = "LIVE"
    buy_usd: float = 7.0
    tp_pct: float = 8.0
    sl_pct: float = 2.0
    soft_sl_pct: float = 1.0
    slow_gain_req: float = 3.0
    stale_minutes: float = 60.0
    universe_pick: str = "auto"

    @classmethod
    def from_env(cls) -> "LiveConfig":
        def f(name: str, default: float) -> float:
            v = os.getenv(name)
            try:
                return float(v) if v is not None else default
            except ValueError:
                return default

        return cls(
            mode=os.getenv("MODE", "LIVE"),
            buy_usd=f("BUY_USD", 7.0),
            tp_pct=f("TP_PCT", 8.0),
            sl_pct=f("SL_PCT", 2.0),
            soft_sl_pct=f("SOFT_SL_PCT", 1.0),
            slow_gain_req=f("SLOW_GAIN_REQ", 3.0),
            stale_minutes=f("STALE_MINUTES", 60.0),
            universe_pick=os.getenv("UNIVERSE_PICK", "auto") or "auto",
        )


def _kraken_private(endpoint: str, data: Optional[Dict[str, str]] = None) -> Dict:
    """
    Minimal private Kraken helper for read-only endpoints (e.g., Balance).
    """
    key = os.getenv("KRAKEN_API_KEY")
    secret = os.getenv("KRAKEN_API_SECRET")

    if not key or not secret:
        raise RuntimeError("Missing KRAKEN_API_KEY / KRAKEN_API_SECRET env vars")

    if data is None:
        data = {}

    path = f"/0/private/{endpoint}"
    url = API_BASE + path

    nonce = str(int(time.time() * 1000))
    data["nonce"] = nonce
    post_data = "&".join(f"{k}={v}" for k, v in data.items())

    # Message for HMAC: path + SHA256(nonce + post_data)
    sha_data = (nonce + post_data).encode("utf-8")
    sha256 = hashlib.sha256(sha_data).digest()
    mac_data = path.encode("utf-8") + sha256

    secret_bytes = base64.b64decode(secret)
    signature = hmac.new(secret_bytes, mac_data, hashlib.sha512)
    api_sign = base64.b64encode(signature.digest()).decode()

    headers = {
        "API-Key": key,
        "API-Sign": api_sign,
        "User-Agent": "live-state-snapshot",
    }

    resp = requests.post(url, headers=headers, data=data, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("error"):
        raise RuntimeError(f"Kraken API error: {payload['error']}")
    return payload["result"]


def fetch_balances() -> Dict[str, float]:
    result = _kraken_private("Balance")
    balances: Dict[str, float] = {}
    for asset, amount in result.items():
        try:
            balances[asset] = float(amount)
        except ValueError:
            continue
    return balances


def pick_primary_coin(balances: Dict[str, float]) -> Optional[tuple[str, float]]:
    """
    Very simple heuristic: choose the first non-USD/USDT/USDC asset with a
    non-trivial balance. This matches our 1-coin-at-a-time usage.
    """
    if not balances:
        return None

    ignore = {"USD", "ZUSD", "USDT", "USDC"}
    # Sort by raw units as a rough proxy for size
    for asset, units in sorted(balances.items(), key=lambda kv: kv[1], reverse=True):
        if units <= 0:
            continue
        if asset in ignore:
            continue
        # Kraken sometimes prefixes spot assets with X/Z (e.g., XXBT, XETH)
        norm = asset
        if norm.startswith(("X", "Z")) and len(norm) > 3:
            norm = norm[1:]
        return norm, units
    return None


def write_positions(symbol: Optional[str], units: float, price: float, now_iso: str) -> None:
    data = {
        "symbol": f"{symbol}/USD" if symbol else None,
        "units": units,
        "entry_price": price,   # here “entry_price” is really “snapshot price”
        "entry_time": now_iso,
    }
    POSITIONS_PATH.write_text(json.dumps(data, indent=2))


def write_summary(cfg: LiveConfig, note_lines: list[str]) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines: list[str] = []
    lines.append("Kraken — 1-Coin Rotation (LIVE snapshot)\n")
    lines.append(f"Timestamp (UTC): {now}")
    lines.append(f"Mode: {cfg.mode}")
    lines.append("Config")
    lines.append(f"  • BUY_USD: {cfg.buy_usd}")
    lines.append(f"  • TP_PCT: {cfg.tp_pct}")
    lines.append(f"  • SL_PCT: {cfg.sl_pct}")
    lines.append(f"  • SOFT_SL_PCT: {cfg.soft_sl_pct}")
    lines.append(f"  • SLOW_GAIN_REQ: {cfg.slow_gain_req}")
    lines.append(f"  • STALE_MINUTES: {cfg.stale_minutes}")
    lines.append(f"  • UNIVERSE_PICK: {cfg.universe_pick}")
    lines.append("")
    lines.append("State / Notes")
    for line in note_lines:
        lines.append(f"  • {line}")
    lines.append("")
    SUMMARY_PATH.write_text("\n".join(lines))


def main() -> None:
    STATE_DIR.mkdir(exist_ok=True)

    cfg = LiveConfig.from_env()
    now = datetime.now(timezone.utc)
    now_iso = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    notes: list[str] = []
    symbol: Optional[str] = None
    units: float = 0.0
    price: float = 0.0

    # 1) Try to read balances from Kraken
    try:
        balances = fetch_balances()
        if not balances:
            notes.append("No balances returned from Kraken.")
        else:
            notes.append(f"Detected {len(balances)} assets in Kraken balance.")
            choice = pick_primary_coin(balances)
            if choice is None:
                notes.append("No non-USD crypto balance detected → treating as FLAT.")
            else:
                symbol, units = choice
                notes.append(f"Primary coin (largest non-USD balance): {symbol} (~{units} units).")
    except Exception as e:
        notes.append(f"ERROR talking to Kraken Balance API: {e!r}")

    # 2) Try to fetch a rough USD price using the public ticker endpoint
    if symbol:
        pair = f"{symbol}/USD"
        try:
            from trader.crypto_engine import get_public_quote  # type: ignore

            price_val = get_public_quote(pair)
            if price_val is None:
                notes.append(f"Could not fetch public price for {pair}; leaving entry_price=0.0.")
            else:
                price = float(price_val)
                notes.append(f"Approx. current price for {pair}: {price}")
        except Exception as e:
            notes.append(f"ERROR fetching public quote for {pair}: {e!r}")

    # 3) Persist snapshot to .state
    try:
        write_positions(symbol, units, price, now_iso)
        notes.append("positions.json updated for LIVE snapshot.")
    except Exception as e:
        notes.append(f"ERROR writing positions.json: {e!r}")

    write_summary(cfg, notes)


if __name__ == "__main__":
    main()
