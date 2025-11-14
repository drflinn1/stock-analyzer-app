#!/usr/bin/env python3
"""
kraken_monday_main.py
Kraken 1-coin rotation bot (Monday Baseline).

Behavior
--------
- Tracks a single symbol (e.g. BTC/USD).
- If FLAT  -> buys BUY_USD worth of that symbol.
- If LONG  -> checks TP_PCT / SL_PCT and exits if hit.
- State saved in .state/kraken_positions.json
- DRY_RUN = ON  -> simulate using market price only.
- DRY_RUN = OFF -> send REAL market orders to Kraken via ccxt.

This is intentionally simple and close to the "good Monday" behavior:
one coin at a time, clear TP/SL, and explicit logging.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ccxt
import requests


STATE_DIR = Path(".state")
STATE_FILE = STATE_DIR / "kraken_positions.json"
SUMMARY_FILE = STATE_DIR / "kraken_run_summary.md"


# ------------------------- Data structures ------------------------- #

@dataclass
class Position:
    symbol: str
    side: str           # "LONG"
    qty: float
    entry_price: float
    opened_at: float    # epoch seconds


@dataclass
class BotConfig:
    symbol: str
    buy_usd: float
    tp_pct: float
    sl_pct: float
    dry_run: bool
    now_ts: float


# ------------------------- Helpers ------------------------- #

def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def get_env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v.strip() if v else default


def get_env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        log(f"WARNING: invalid float for {name}={v!r}, using default {default}")
        return default


def get_config_from_env() -> BotConfig:
    symbol = get_env_str("SYMBOL", "BTC/USD")
    buy_usd = get_env_float("BUY_USD", 10.0)
    tp_pct = get_env_float("TP_PCT", 5.0)
    sl_pct = get_env_float("SL_PCT", 2.0)
    dry_run = get_env_str("DRY_RUN", "ON").upper() != "OFF"
    return BotConfig(
        symbol=symbol,
        buy_usd=buy_usd,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        dry_run=dry_run,
        now_ts=time.time(),
    )


def load_position() -> Optional[Position]:
    if not STATE_FILE.exists():
        return None
    try:
        data = json.loads(STATE_FILE.read_text())
        return Position(
            symbol=data["symbol"],
            side=data["side"],
            qty=float(data["qty"]),
            entry_price=float(data["entry_price"]),
            opened_at=float(data["opened_at"]),
        )
    except Exception as e:
        log(f"WARNING: failed to load position file: {e}")
        return None


def save_position(pos: Optional[Position]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if pos is None:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        return
    STATE_FILE.write_text(json.dumps(asdict(pos), indent=2))


def write_summary(lines: list[str]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text("\n".join(lines))


# ------------------------- Market data ------------------------- #

def fetch_spot_price(symbol: str) -> Optional[float]:
    """
    Fetch spot price from Kraken via ccxt public API.
    ccxt normalizes Kraken symbols, so BTC/USD should work (it maps to XBT/USD).
    """
    try:
        ex = ccxt.kraken()
        ticker = ex.fetch_ticker(symbol)
        price = ticker["last"]
        if price is None:
            raise ValueError("no last price in ticker")
        return float(price)
    except Exception as e:
        log(f"ERROR: Failed to fetch spot price for {symbol}: {e}")
        return None


# ------------------------- Trading actions ------------------------- #

def simulate_buy(cfg: BotConfig, price: float) -> Position:
    qty = cfg.buy_usd / price if price > 0 else 0.0
    qty = round(qty, 6)
    log(f"[DRY_RUN] BUY {cfg.symbol} for ${cfg.buy_usd:.2f} @ ${price:.2f} (qty ~ {qty})")
    return Position(
        symbol=cfg.symbol,
        side="LONG",
        qty=qty,
        entry_price=price,
        opened_at=cfg.now_ts,
    )


def simulate_sell(cfg: BotConfig, pos: Position, price: float, reason: str) -> None:
    pnl = (price - pos.entry_price) * pos.qty
    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100.0
    log(
        f"[DRY_RUN] SELL {cfg.symbol} {pos.qty} @ ${price:.2f} "
        f"(entry ${pos.entry_price:.2f}, PnL ${pnl:.2f}, {pnl_pct:.2f}%, reason={reason})"
    )


def get_authenticated_exchange() -> Optional[ccxt.kraken]:
    key = os.getenv("KRAKEN_API_KEY")
    secret = os.getenv("KRAKEN_API_SECRET")
    if not key or not secret:
        log("ERROR: KRAKEN_API_KEY / KRAKEN_API_SECRET not set.")
        return None
    try:
        ex = ccxt.kraken({
            "apiKey": key,
            "secret": secret,
        })
        ex.load_markets()
        return ex
    except Exception as e:
        log(f"ERROR: Failed to create authenticated Kraken client: {e}")
        return None


def real_buy(cfg: BotConfig, price: float) -> Optional[Position]:
    ex = get_authenticated_exchange()
    if ex is None:
        return None

    # Compute quantity in base currency
    qty = cfg.buy_usd / price if price > 0 else 0.0
    # Round to something reasonable to avoid Kraken min-precision issues
    qty = float(ex.amount_to_precision(cfg.symbol, qty))

    log(f"Placing REAL market BUY {cfg.symbol} for ${cfg.buy_usd:.2f} (qty={qty})")

    try:
        order = ex.create_market_buy_order(cfg.symbol, qty)
    except Exception as e:
        log(f"ERROR: Kraken BUY failed: {e}")
        return None

    try:
        log(f"Kraken BUY response: {order}")
    except Exception:
        log("Kraken BUY response received (could not pretty-print).")

    pos = Position(
        symbol=cfg.symbol,
        side="LONG",
        qty=qty,
        entry_price=price,
        opened_at=cfg.now_ts,
    )
    log(f"REAL BUY OK → {cfg.symbol} qty={qty} @ ${price:.2f}")
    return pos


def real_sell(cfg: BotConfig, pos: Position, price: float, reason: str) -> None:
    ex = get_authenticated_exchange()
    if ex is None:
        return

    qty = float(ex.amount_to_precision(cfg.symbol, pos.qty))

    log(
        f"Placing REAL market SELL {cfg.symbol} qty={qty} "
        f"(entry={pos.entry_price:.2f}, now={price:.2f}, reason={reason})"
    )

    try:
        order = ex.create_market_sell_order(cfg.symbol, qty)
    except Exception as e:
        log(f"ERROR: Kraken SELL failed: {e}")
        return

    try:
        log(f"Kraken SELL response: {order}")
    except Exception:
        log("Kraken SELL response received (could not pretty-print).")

    pnl = (price - pos.entry_price) * pos.qty
    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100.0
    log(f"REAL SELL OK → PnL ${pnl:.2f} ({pnl_pct:.2f}%)")


# ------------------------- Main logic ------------------------- #

def run() -> int:
    cfg = get_config_from_env()
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Kraken 1-Coin Rotation Run (Monday Baseline)")
    lines.append(f"- Time: {datetime.fromtimestamp(cfg.now_ts, tz=timezone.utc)}")
    lines.append(f"- Symbol: {cfg.symbol}")
    lines.append(f"  BUY_USD: {cfg.buy_usd}")
    lines.append(f"  TP_PCT: {cfg.tp_pct}")
    lines.append(f"  SL_PCT: {cfg.sl_pct}")
    lines.append(f"  DRY_RUN: {'ON' if cfg.dry_run else 'OFF'}")
    lines.append("")

    log("Starting Kraken 1-coin rotation run...")
    log(
        f"Config: symbol={cfg.symbol}, BUY_USD={cfg.buy_usd}, "
        f"TP_PCT={cfg.tp_pct}, SL_PCT={cfg.sl_pct}, DRY_RUN={'ON' if cfg.dry_run else 'OFF'}"
    )

    price = fetch_spot_price(cfg.symbol)
    if price is None:
        lines.append("- ERROR: Failed to fetch spot price; aborting.")
        write_summary(lines)
        return 1

    log(f"Current spot price {cfg.symbol} = ${price:.2f}")
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
            f"- Action: BUY -> {cfg.symbol}, qty ~ {new_pos.qty}, "
            f"entry_price=${new_pos.entry_price:.2f}"
        )
        write_summary(lines)
        log("Run complete (BUY).")
        return 0

    # We have a position -> evaluate TP/SL
    log(
        f"Existing position: {pos.symbol} qty={pos.qty}, "
        f"entry_price=${pos.entry_price:.2f}, "
        f"opened_at={datetime.fromtimestamp(pos.opened_at, tz=timezone.utc)}"
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
        log("HOLD: TP/SL not hit. No action taken.")
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
