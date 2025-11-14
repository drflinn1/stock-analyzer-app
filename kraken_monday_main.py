#!/usr/bin/env python3
"""
kraken_monday_main.py

Kraken 1-Coin Rotation (Monday Baseline) with DRY and LIVE modes.

DRY_RUN = "ON"
    - Simulate BTC/USD trades using local state only.
    - No Kraken private API calls.
    - Uses public ticker for price.
    - Writes markdown + JSON so we can inspect behaviour safely.

DRY_RUN = "OFF"
    - Uses Kraken private API (requires KRAKEN_API_KEY / KRAKEN_API_SECRET).
    - Checks real BTC balance to decide if we are "in position".
    - Uses local state file for entry_price / qty and syncs if needed.
    - Applies TP_PCT / SL_PCT (take profit / stop loss).
    - Sends real MARKET BUY/SELL orders.
    - Writes markdown + JSON summary of every decision.

Config via environment variables (all optional):

    SYMBOL          (default "BTC/USD")
    BUY_USD         (default "20")      # USD notional for new buys
    TP_PCT          (default "8")       # take profit %
    SL_PCT          (default "1")       # stop loss %
    DRY_RUN         (default "ON")      # "ON" or "OFF"

    KRAKEN_API_KEY      (required for DRY_RUN="OFF")
    KRAKEN_API_SECRET   (required for DRY_RUN="OFF")

Files written under .state/:

    .state/kraken_monday_state.json  # persistent position state
    .state/kraken_monday_run.json    # one-run summary (machine-readable)
    .state/kraken_monday_run.md      # pretty markdown summary (for artifacts)

Safe to run repeatedly; each run will:
    - Fetch price
    - Decide BUY / SELL / HOLD
    - Execute behaviour depending on DRY_RUN
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from urllib.parse import urlencode

# ---------- Basic config ----------

STATE_DIR = Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_JSON = STATE_DIR / "kraken_monday_state.json"
RUN_JSON = STATE_DIR / "kraken_monday_run.json"
RUN_MD = STATE_DIR / "kraken_monday_run.md"

KRAKEN_API_BASE = "https://api.kraken.com"
KRAKEN_API_VERSION = "0"

KRAKEN_MIN_NOTIONAL_USD = 10.0  # conservative min to avoid 'EOrder:Insufficient funds'


# ---------- Helper dataclasses ----------

@dataclass
class PositionState:
    symbol: str
    status: str  # "flat" or "in_position"
    qty: float
    entry_price: float
    last_action: str
    last_run: str

    @classmethod
    def empty(cls, symbol: str) -> "PositionState":
        now = dt.datetime.utcnow().isoformat()
        return cls(
            symbol=symbol,
            status="flat",
            qty=0.0,
            entry_price=0.0,
            last_action="INIT_FLAT",
            last_run=now,
        )


@dataclass
class RunSummary:
    time: str
    symbol: str
    buy_usd: float
    tp_pct: float
    sl_pct: float
    dry_run: str
    spot_price: float
    status: str
    action: str
    pnl_pct: Optional[float] = None
    notes: Optional[str] = None


# ---------- Utility functions ----------

def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def load_state(symbol: str) -> PositionState:
    if not STATE_JSON.exists():
        return PositionState.empty(symbol)
    try:
        data = json.loads(STATE_JSON.read_text())
        return PositionState(
            symbol=data.get("symbol", symbol),
            status=data.get("status", "flat"),
            qty=float(data.get("qty", 0.0)),
            entry_price=float(data.get("entry_price", 0.0)),
            last_action=data.get("last_action", "INIT_LOAD"),
            last_run=data.get("last_run", dt.datetime.utcnow().isoformat()),
        )
    except Exception:
        # Corrupt state? Start flat but keep file around.
        return PositionState.empty(symbol)


def save_state(state: PositionState) -> None:
    STATE_JSON.write_text(json.dumps(asdict(state), indent=2))


def write_run_outputs(summary: RunSummary) -> None:
    # JSON
    RUN_JSON.write_text(json.dumps(asdict(summary), indent=2))

    # Markdown
    lines = [
        "# Kraken 1-Coin Rotation Run (Monday Baseline)",
        "",
        f"- Time: {summary.time}",
        f"- Symbol: {summary.symbol}",
        f"  BUY_USD: {summary.buy_usd}",
        f"  TP_PCT: {summary.tp_pct}",
        f"  SL_PCT: {summary.sl_pct}",
        f"  DRY_RUN: {summary.dry_run}",
        f"- Spot price: ${summary.spot_price:.2f}",
        f"- Status: {summary.status}",
        f"- Action: {summary.action}",
    ]
    if summary.pnl_pct is not None:
        lines.append(f"- PnL: {summary.pnl_pct:.2f}%")
    if summary.notes:
        lines.append(f"- Notes: {summary.notes}")
    RUN_MD.write_text("\n".join(lines))

    # Also print to stdout so it shows in logs
    print("\n".join(lines))


def kraken_public_ticker_btcusd() -> float:
    """
    Fetches spot price for BTC/USD using Kraken's public ticker.

    Kraken uses XBTUSD as the pair name; we grab the first result's c[0].
    """
    url = f"{KRAKEN_API_BASE}/{KRAKEN_API_VERSION}/public/Ticker"
    resp = requests.get(url, params={"pair": "XBTUSD"}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken public error: {data['error']}")
    result = data["result"]
    # Just pick the first key (usually "XXBTZUSD")
    first_key = next(iter(result.keys()))
    last_str = result[first_key]["c"][0]  # last trade price
    return float(last_str)


def kraken_private_request(
    path: str,
    data: Dict[str, Any],
    api_key: str,
    api_secret_b64: str,
) -> Dict[str, Any]:
    """
    Minimal Kraken authenticated POST request.

    path: e.g. "/0/private/Balance"
    """
    nonce = str(int(time.time() * 1000))
    data = {"nonce": nonce, **data}
    postdata = urlencode(data)
    # Message for HMAC: path + SHA256(nonce + postdata)
    sha256 = hashlib.sha256((nonce + postdata).encode()).digest()
    message = path.encode() + sha256
    secret = base64.b64decode(api_secret_b64)
    sig = hmac.new(secret, message, hashlib.sha512)
    sig_b64 = base64.b64encode(sig.digest()).decode()

    headers = {
        "API-Key": api_key,
        "API-Sign": sig_b64,
    }

    url = f"{KRAKEN_API_BASE}{path}"
    resp = requests.post(url, data=data, headers=headers, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("error"):
        raise RuntimeError(f"Kraken private error: {payload['error']}")
    return payload["result"]


def kraken_get_btc_balance(api_key: str, api_secret_b64: str) -> float:
    """
    Return free BTC balance (XBT) on Kraken spot account.
    """
    result = kraken_private_request(
        f"/{KRAKEN_API_VERSION}/private/Balance",
        {},
        api_key,
        api_secret_b64,
    )
    # Spot BTC is usually under "XXBT" or "XBT"
    for key in ("XXBT", "XBT"):
        if key in result:
            return float(result[key])
    return 0.0


def kraken_market_order(
    side: str,
    qty: float,
    api_key: str,
    api_secret_b64: str,
) -> str:
    """
    Place a market order on BTC/USD (XBTUSD).

    Returns: order description string from Kraken.
    """
    if qty <= 0:
        raise ValueError("qty must be > 0 for live order")

    result = kraken_private_request(
        f"/{KRAKEN_API_VERSION}/private/AddOrder",
        {
            "pair": "XBTUSD",
            "type": side,           # "buy" or "sell"
            "ordertype": "market",
            "volume": f"{qty:.8f}",
        },
        api_key,
        api_secret_b64,
    )
    descr = result.get("descr", {})
    order_descr = descr.get("order", "order sent")
    return order_descr


# ---------- Core logic ----------

def run() -> None:
    # Read config
    symbol = os.getenv("SYMBOL", "BTC/USD")
    buy_usd = env_float("BUY_USD", 20.0)
    tp_pct = env_float("TP_PCT", 8.0)
    sl_pct = env_float("SL_PCT", 1.0)
    dry_run = os.getenv("DRY_RUN", "ON").strip().upper() or "ON"

    now = dt.datetime.utcnow().isoformat()

    # Load state
    state = load_state(symbol)

    # Fetch spot price
    try:
        price = kraken_public_ticker_btcusd()
    except Exception as e:
        # If we can't get a price, fail safely.
        summary = RunSummary(
            time=now,
            symbol=symbol,
            buy_usd=buy_usd,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            dry_run=dry_run,
            spot_price=0.0,
            status=state.status,
            action="HOLD (no price available)",
            notes=f"Error fetching price: {e}",
        )
        write_run_outputs(summary)
        return

    # Determine if we are in position (base on state +, in LIVE, wallet)
    api_key = os.getenv("KRAKEN_API_KEY", "").strip()
    api_secret = os.getenv("KRAKEN_API_SECRET", "").strip()

    live_btc_balance = None
    notes: list[str] = []

    if dry_run == "OFF":
        if not api_key or not api_secret:
            summary = RunSummary(
                time=now,
                symbol=symbol,
                buy_usd=buy_usd,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                dry_run=dry_run,
                spot_price=price,
                status=state.status,
                action="HOLD (missing API keys)",
                notes="KRAKEN_API_KEY / KRAKEN_API_SECRET not set; refusing to trade.",
            )
            write_run_outputs(summary)
            return

        try:
            live_btc_balance = kraken_get_btc_balance(api_key, api_secret)
            notes.append(f"Live BTC balance: {live_btc_balance:.8f}")
        except Exception as e:
            summary = RunSummary(
                time=now,
                symbol=symbol,
                buy_usd=buy_usd,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                dry_run=dry_run,
                spot_price=price,
                status=state.status,
                action="HOLD (error reading balance)",
                notes=f"Error reading Kraken balance: {e}",
            )
            write_run_outputs(summary)
            return

        # Sync state with live wallet if they disagree materially
        if live_btc_balance and live_btc_balance > 0.00001 and state.status == "flat":
            # We have BTC on Kraken but state thinks we're flat — sync in.
            state.status = "in_position"
            state.qty = live_btc_balance
            state.entry_price = price  # unknown true basis; use current spot
            state.last_action = "SYNC_IN_POSITION_FROM_WALLET"
            notes.append(
                "State was flat but live BTC balance > 0; "
                "synced state to IN_POSITION using current price as entry."
            )
        elif (not live_btc_balance or live_btc_balance <= 0.00001) and state.status == "in_position":
            # State thought we were in, but wallet has no BTC — sync flat.
            state.status = "flat"
            state.qty = 0.0
            state.entry_price = 0.0
            state.last_action = "SYNC_FLAT_FROM_WALLET"
            notes.append(
                "State was IN_POSITION but live BTC balance ~ 0; "
                "synced state to FLAT."
            )

    # PnL if in position
    pnl_pct = None
    if state.status == "in_position" and state.entry_price > 0:
        pnl_pct = (price - state.entry_price) / state.entry_price * 100.0

    # Decide action
    action = "HOLD"
    if state.status == "in_position":
        # Check TP / SL
        if pnl_pct is not None and pnl_pct >= tp_pct:
            action = f"SELL (TP hit: {pnl_pct:.2f}% >= {tp_pct}%)"
        elif pnl_pct is not None and pnl_pct <= -sl_pct:
            action = f"SELL (SL hit: {pnl_pct:.2f}% <= -{sl_pct}%)"
        else:
            action = "HOLD (in position; TP/SL not hit)"
    else:
        # Flat — decide if we can buy
        notional = buy_usd
        if notional < KRAKEN_MIN_NOTIONAL_USD:
            action = (
                f"HOLD (flat; BUY_USD={buy_usd} below Kraken min "
                f"{KRAKEN_MIN_NOTIONAL_USD})"
            )
        else:
            qty = buy_usd / price
            action = f"BUY -> {symbol}, qty ~ {qty:.6f}, entry_price=${price:.2f}"

    # Execute behaviour based on DRY_RUN
    if dry_run == "ON":
        # Simulated behaviour
        if action.startswith("BUY ->"):
            qty = buy_usd / price
            state.status = "in_position"
            state.qty = qty
            state.entry_price = price
            state.last_action = "SIM_BUY"
        elif action.startswith("SELL (TP hit") or action.startswith("SELL (SL hit"):
            # Simulated sell to flat
            state.status = "flat"
            state.qty = 0.0
            state.entry_price = 0.0
            state.last_action = "SIM_SELL"
        else:
            state.last_action = "SIM_HOLD"

    else:
        # LIVE trading
        if action.startswith("BUY ->"):
            qty = buy_usd / price
            try:
                order_descr = kraken_market_order(
                    side="buy",
                    qty=qty,
                    api_key=api_key,
                    api_secret_b64=api_secret,
                )
                notes.append(f"BUY order sent: {order_descr}")
                state.status = "in_position"
                state.qty = qty
                state.entry_price = price
                state.last_action = "LIVE_BUY"
            except Exception as e:
                notes.append(f"BUY failed: {e}")
                action = f"HOLD (BUY failed: {e})"
                state.last_action = "LIVE_BUY_FAILED"

        elif action.startswith("SELL (TP hit") or action.startswith("SELL (SL hit"):
            # Sell all BTC we think we hold
            qty_to_sell = state.qty
            if live_btc_balance is not None:
                qty_to_sell = max(qty_to_sell, live_btc_balance)
            if qty_to_sell <= 0.00001:
                notes.append("No BTC to sell; skipping SELL.")
                state.last_action = "LIVE_SELL_SKIPPED_NO_BALANCE"
                action = "HOLD (no BTC to sell)"
            else:
                try:
                    order_descr = kraken_market_order(
                        side="sell",
                        qty=qty_to_sell,
                        api_key=api_key,
                        api_secret_b64=api_secret,
                    )
                    notes.append(f"SELL order sent: {order_descr}")
                    state.status = "flat"
                    state.qty = 0.0
                    state.entry_price = 0.0
                    state.last_action = "LIVE_SELL"
                except Exception as e:
                    notes.append(f"SELL failed: {e}")
                    action = f"HOLD (SELL failed: {e})"
                    state.last_action = "LIVE_SELL_FAILED"
        else:
            state.last_action = "LIVE_HOLD"

    state.last_run = now
    save_state(state)

    summary = RunSummary(
        time=now,
        symbol=symbol,
        buy_usd=buy_usd,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        dry_run=dry_run,
        spot_price=price,
        status=state.status,
        action=action,
        pnl_pct=pnl_pct,
        notes=" | ".join(notes) if notes else None,
    )
    write_run_outputs(summary)


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        # Fail safely, with something in the artifacts/logs.
        now = dt.datetime.utcnow().isoformat()
        err_summary = RunSummary(
            time=now,
            symbol=os.getenv("SYMBOL", "BTC/USD"),
            buy_usd=env_float("BUY_USD", 20.0),
            tp_pct=env_float("TP_PCT", 8.0),
            sl_pct=env_float("SL_PCT", 1.0),
            dry_run=os.getenv("DRY_RUN", "ON").strip().upper() or "ON",
            spot_price=0.0,
            status="error",
            action="HOLD (exception)",
            notes=f"Unhandled error: {exc}",
        )
        write_run_outputs(err_summary)
        # Also send traceback to logs
        print("Unhandled exception:", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
