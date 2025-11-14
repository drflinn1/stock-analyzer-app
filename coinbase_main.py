#!/usr/bin/env python3
"""
coinbase_main.py
Simple 1-coin rotation bot for Coinbase Advanced.

Behavior
--------
- Tracks a single product (e.g. BTC-USD).
- If flat -> buys using BUY_USD.
- If long  -> checks TP_PCT / SL_PCT and exits if hit.
- State is stored in .state/coinbase_positions.json
- DRY_RUN = ON  -> simulate buys/sells using public spot price.
- DRY_RUN = OFF -> use Coinbase Advanced API Python SDK to place real orders
                   (requires COINBASE_API_KEY and COINBASE_API_SECRET).

This is intentionally simple and safe. We can add momentum-scanning and
multi-coin rotation later once Coinbase is fully stable.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests  # Used for public spot price (no auth required)

STATE_DIR = Path(".state")
STATE_FILE = STATE_DIR / "coinbase_positions.json"
SUMMARY_FILE = STATE_DIR / "coinbase_run_summary.md"


# ------------------------- Data structures ------------------------- #

@dataclass
class Position:
    product_id: str
    side: str           # "LONG" for now
    qty: float
    entry_price: float
    opened_at: float    # epoch seconds


@dataclass
class BotConfig:
    product_id: str
    buy_usd: float
    tp_pct: float
    sl_pct: float
    dry_run: bool
    now_ts: float


# ------------------------- Helpers ------------------------- #

def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def load_position() -> Optional[Position]:
    if not STATE_FILE.exists():
        return None
    try:
        data = json.loads(STATE_FILE.read_text())
        return Position(
            product_id=data["product_id"],
            side=data["side"],
            qty=float(data["qty"]),
            entry_price=float(data["entry_price"]),
            opened_at=float(data["opened_at"]),
        )
    except Exception as e:
        log(f"WARNING: Failed to load position file: {e}")
        return None


def save_position(pos: Optional[Position]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if pos is None:
        # Clear file
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        return
    STATE_FILE.write_text(json.dumps(asdict(pos), indent=2))


def write_summary(lines: list[str]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text("\n".join(lines))


def get_env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        log(f"WARNING: invalid float for {name}={val!r}, using default {default}")
        return default


def get_env_str(name: str, default: str) -> str:
    val = os.getenv(name)
    return val.strip() if val else default


def get_config_from_env() -> BotConfig:
    product_id = get_env_str("SYMBOL", "BTC-USD")
    buy_usd = get_env_float("BUY_USD", 10.0)
    tp_pct = get_env_float("TP_PCT", 5.0)
    sl_pct = get_env_float("SL_PCT", 2.0)
    dry = get_env_str("DRY_RUN", "ON").upper() != "OFF"
    return BotConfig(
        product_id=product_id,
        buy_usd=buy_usd,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        dry_run=dry,
        now_ts=time.time(),
    )


# ------------------------- Market data ------------------------- #

def fetch_spot_price(product_id: str) -> Optional[float]:
    """
    Uses public Coinbase v2 spot endpoint.
    Example: GET https://api.coinbase.com/v2/prices/BTC-USD/spot
    No auth required.
    """
    url = f"https://api.coinbase.com/v2/prices/{product_id}/spot"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        amt = float(data["data"]["amount"])
        return amt
    except Exception as e:
        log(f"ERROR: Failed to fetch spot price for {product_id}: {e}")
        return None


# ------------------------- Trading actions ------------------------- #

def simulate_buy(cfg: BotConfig, price: float) -> Position:
    qty = cfg.buy_usd / price if price > 0 else 0.0
    qty = round(qty, 8)
    log(f"[DRY_RUN] BUY {cfg.product_id} for ${cfg.buy_usd:.2f} @ ${price:.2f} (qty ~ {qty})")
    return Position(
        product_id=cfg.product_id,
        side="LONG",
        qty=qty,
        entry_price=price,
        opened_at=cfg.now_ts,
    )


def simulate_sell(cfg: BotConfig, pos: Position, price: float, reason: str) -> None:
    pnl = (price - pos.entry_price) * pos.qty
    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100.0
    log(
        f"[DRY_RUN] SELL {cfg.product_id} {pos.qty} @ ${price:.2f} "
        f"(entry ${pos.entry_price:.2f}, PnL ${pnl:.2f}, {pnl_pct:.2f}%, reason={reason})"
    )


def real_buy(cfg: BotConfig, price: float) -> Optional[Position]:
    """
    Use Coinbase Advanced RESTClient to place a market buy.
    Requires COINBASE_API_KEY and COINBASE_API_SECRET.
    """
    api_key = os.getenv("COINBASE_API_KEY")
    api_secret = os.getenv("COINBASE_API_SECRET")

    if not api_key or not api_secret:
        log("ERROR: COINBASE_API_KEY / COINBASE_API_SECRET not set; cannot place real BUY.")
        return None

    try:
        from coinbase.rest import RESTClient  # type: ignore
    except Exception as e:
        log(f"ERROR: Failed to import coinbase RESTClient: {e}")
        return None

    client = RESTClient(api_key=api_key, api_secret=api_secret)

    quote_size = f"{cfg.buy_usd:.2f}"
    client_order_id = f"cb-buy-{int(cfg.now_ts)}"

    log(f"Placing REAL market BUY {cfg.product_id} for ${quote_size} (client_order_id={client_order_id})")

    try:
        order = client.market_order_buy(
            client_order_id=client_order_id,
            product_id=cfg.product_id,
            quote_size=quote_size,
        )
    except Exception as e:
        log(f"ERROR: Coinbase BUY failed: {e}")
        return None

    # Parse response
    success = bool(order.get("success"))
    if not success:
        log(f"ERROR: Coinbase BUY not successful: {order.get('error_response')}")
        return None

    sr = order.get("success_response", {}) or {}
    order_id = sr.get("order_id")
    log(f"BUY order placed successfully, order_id={order_id}")

    # We don't have fill info here in the simple path; use current price as entry
    qty = cfg.buy_usd / price if price > 0 else 0.0
    qty = round(qty, 8)
    return Position(
        product_id=cfg.product_id,
        side="LONG",
        qty=qty,
        entry_price=price,
        opened_at=cfg.now_ts,
    )


def real_sell(cfg: BotConfig, pos: Position, price: float, reason: str) -> None:
    api_key = os.getenv("COINBASE_API_KEY")
    api_secret = os.getenv("COINBASE_API_SECRET")

    if not api_key or not api_secret:
        log("ERROR: COINBASE_API_KEY / COINBASE_API_SECRET not set; cannot place real SELL.")
        return

    try:
        from coinbase.rest import RESTClient  # type: ignore
    except Exception as e:
        log(f"ERROR: Failed to import coinbase RESTClient: {e}")
        return

    client = RESTClient(api_key=api_key, api_secret=api_secret)
    client_order_id = f"cb-sell-{int(cfg.now_ts)}"
    base_size = f"{pos.qty:.8f}"

    log(
        f"Placing REAL market SELL {cfg.product_id} qty {base_size} "
        f"(reason={reason}, entry={pos.entry_price:.2f}, now={price:.2f})"
    )

    try:
        order = client.market_order_sell(
            client_order_id=client_order_id,
            product_id=cfg.product_id,
            base_size=base_size,
        )
    except Exception as e:
        log(f"ERROR: Coinbase SELL failed: {e}")
        return

    success = bool(order.get("success"))
    if not success:
        log(f"ERROR: Coinbase SELL not successful: {order.get('error_response')}")
        return

    sr = order.get("success_response", {}) or {}
    order_id = sr.get("order_id")
    pnl = (price - pos.entry_price) * pos.qty
    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100.0

    log(
        f"SELL order placed successfully, order_id={order_id}, "
        f"PnL ${pnl:.2f}, {pnl_pct:.2f}%"
    )


# ------------------------- Main logic ------------------------- #

def run() -> int:
    cfg = get_config_from_env()
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# Coinbase 1-Coin Rotation Run")
    lines.append(f"- Time: {datetime.fromtimestamp(cfg.now_ts, tz=timezone.utc)}")
    lines.append(f"- Product: {cfg.product_id}")
    lines.append(f"- BUY_USD: {cfg.buy_usd}")
    lines.append(f"- TP_PCT: {cfg.tp_pct}")
    lines.append(f"- SL_PCT: {cfg.sl_pct}")
    lines.append(f"- DRY_RUN: {'ON' if cfg.dry_run else 'OFF'}")
    lines.append("")

    log("Starting Coinbase 1-coin rotation run...")
    log(f"Config: product_id={cfg.product_id}, BUY_USD={cfg.buy_usd}, "
        f"TP_PCT={cfg.tp_pct}, SL_PCT={cfg.sl_pct}, DRY_RUN={'ON' if cfg.dry_run else 'OFF'}")

    price = fetch_spot_price(cfg.product_id)
    if price is None:
        lines.append("- ERROR: Failed to fetch spot price; aborting.")
        write_summary(lines)
        return 1

    log(f"Current spot price {cfg.product_id} = ${price:.2f}")
    lines.append(f"- Spot price: ${price:.2f}")

    pos = load_position()
    if pos is None:
        # Flat -> BUY
        log("No existing position -> we are FLAT.")
        if cfg.dry_run:
            new_pos = simulate_buy(cfg, price)
        else:
            new_pos = real_buy(cfg, price)
            if new_pos is None:
                lines.append("- ERROR: Real BUY failed; staying flat.")
                write_summary(lines)
                return 1

        save_position(new_pos)
        lines.append(
            f"- Action: BUY -> {cfg.product_id}, qty ~ {new_pos.qty}, "
            f"entry_price=${new_pos.entry_price:.2f}"
        )
        write_summary(lines)
        log("Run complete (BUY).")
        return 0

    # We have a position -> evaluate TP/SL
    log(
        f"Existing position: {pos.product_id} qty={pos.qty}, "
        f"entry_price=${pos.entry_price:.2f}, opened_at={datetime.fromtimestamp(pos.opened_at, tz=timezone.utc)}"
    )

    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100.0
    pnl_usd = (price - pos.entry_price) * pos.qty
    age_min = (cfg.now_ts - pos.opened_at) / 60.0

    log(f"PnL: {pnl_usd:.2f} USD ({pnl_pct:.2f}%), age={age_min:.1f} min")
    lines.append(f"- Existing position PnL: ${pnl_usd:.2f} ({pnl_pct:.2f}%), age={age_min:.1f} min")

    reason = None
    if pnl_pct >= cfg.tp_pct:
        reason = f"TP hit ({pnl_pct:.2f}% >= {cfg.tp_pct}%)"
    elif pnl_pct <= -cfg.sl_pct:
        reason = f"SL hit ({pnl_pct:.2f}% <= -{cfg.sl_pct}%)"

    if reason is None:
        log("Hold: TP/SL not hit. No action taken.")
        lines.append("- Action: HOLD (TP/SL not hit)")
        write_summary(lines)
        return 0

    # Need to SELL
    if cfg.dry_run:
        simulate_sell(cfg, pos, price, reason)
    else:
        real_sell(cfg, pos, price, reason)

    save_position(None)
    lines.append(f"- Action: SELL -> reason={reason}")
    lines.append(f"- Exit price: ${price:.2f}")
    lines.append(f"- Realized PnL: ${pnl_usd:.2f} ({pnl_pct:.2f}%)")

    write_summary(lines)
    log("Run complete (SELL).")
    return 0


if __name__ == "__main__":
    sys.exit(run())
