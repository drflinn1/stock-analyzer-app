#!/usr/bin/env python3
"""
alpaca_crypto_paper.py
Single-file paper-trading bot for Alpaca Crypto.

Behavior (one-shot per run):
- Checks current position in SYMBOL (default: BTC/USD).
- If no position → market BUY (notional = BUY_USD).
- If holding → evaluate TP/SL and SELL all if hit.
- Emits a human-readable summary + writes .state/run_summary.json.

Env Vars:
  ALPACA_API_KEY         (required)
  ALPACA_API_SECRET      (required)
  SYMBOL                 default "BTC/USD"
  BUY_USD                default "25"  (USD notional per entry)
  TP_PCT                 default "5"   (take-profit %, e.g., 5 = +5%)
  SL_PCT                 default "2"   (stop-loss %, e.g., 2 = -2%)
  DRY_RUN                default "OFF" ("ON" = simulate orders only)
  MAX_RETRY              default "3"
"""

from __future__ import annotations
import json, os, sys, time
from pathlib import Path
from typing import Optional

# Alpaca SDK (modern)
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.common import APIError

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoLatestQuoteRequest

STATE_DIR = Path(".state")
STATE_DIR.mkdir(exist_ok=True)

def getenv(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None or str(v).strip() == "":
        return "" if default is None else str(default)
    return str(v).strip()

def to_float(s: str, default: float) -> float:
    try:
        return float(str(s).strip())
    except Exception:
        return default

def write_json(p: Path, data: dict):
    try:
        p.write_text(json.dumps(data, indent=2, sort_keys=True))
    except Exception as e:
        print(f"WARN: failed to write {p}: {e}", file=sys.stderr)

def mid_from_quote(quote) -> Optional[float]:
    try:
        bid = float(quote.bid_price)
        ask = float(quote.ask_price)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
    except Exception:
        pass
    return None

def main() -> int:
    api_key = getenv("ALPACA_API_KEY")
    api_secret = getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        print("ERROR: Missing ALPACA_API_KEY or ALPACA_API_SECRET")
        return 1

    symbol = getenv("SYMBOL", "BTC/USD")
    buy_usd = to_float(getenv("BUY_USD", "25"), 25.0)
    tp_pct  = to_float(getenv("TP_PCT", "5"), 5.0)
    sl_pct  = to_float(getenv("SL_PCT", "2"), 2.0)
    dry_run = getenv("DRY_RUN", "OFF").upper() == "ON"
    max_retry = int(to_float(getenv("MAX_RETRY", "3"), 3))

    # Init clients (paper=True guarantees paper env)
    trading = TradingClient(api_key, api_secret, paper=True)
    data_cli = CryptoHistoricalDataClient()

    # Latest quote for context
    latest_quote = None
    mid_price = None
    try:
        req = CryptoLatestQuoteRequest(symbol_or_symbols=[symbol])
        resp = data_cli.get_crypto_latest_quote(req)
        latest_quote = resp.get(symbol)
        if latest_quote:
            mid_price = mid_from_quote(latest_quote)
    except Exception as e:
        print(f"WARN: quote fetch failed: {e}")

    # Load positions and find this symbol (Alpaca uses slash format for crypto, e.g., BTC/USD)
    pos = None
    try:
        positions = trading.get_all_positions()
        for p in positions:
            if p.symbol.upper() == symbol.upper():
                pos = p
                break
    except APIError as e:
        print(f"ERROR: fetching positions: {e}")
        return 1

    action = "HOLD"
    order_id = None
    note = ""

    def place_market(side: OrderSide, notional_usd: Optional[float]=None, qty: Optional[float]=None):
        nonlocal order_id
        if dry_run:
            print(f"DRY_RUN → would place {side.name} {symbol} "
                  f"{'(notional=$%.2f)'%notional_usd if notional_usd else ''}"
                  f"{'(qty=%s)'%qty if qty else ''}")
            return

        req_kwargs = dict(symbol=symbol, time_in_force=TimeInForce.GTC)
        if notional_usd is not None:
            req_kwargs["notional"] = float(notional_usd)
        elif qty is not None:
            req_kwargs["qty"] = float(qty)
        else:
            raise ValueError("Either notional_usd or qty must be provided")

        mo = MarketOrderRequest(side=side, **req_kwargs)
        retry = 0
        while True:
            try:
                o = trading.submit_order(order_data=mo)
                order_id = o.id
                break
            except APIError as e:
                retry += 1
                if retry > max_retry:
                    raise
                print(f"Order submit retry {retry}/{max_retry} after APIError: {e}")
                time.sleep(2)

    if pos is None:
        # no position → BUY
        try:
            action = "BUY"
            place_market(OrderSide.BUY, notional_usd=buy_usd)
            note = f"Entered {symbol} with ${buy_usd:.2f} notional."
        except APIError as e:
            action = "ERROR"
            note = f"BUY failed: {e}"
            print(f"ERROR: {note}")
    else:
        # Evaluate TP/SL using unrealized P&L percent (plpc is decimal, e.g., 0.051 = +5.1%)
        try:
            # pos.unrealized_plpc can be None if flat, guard it
            plpc = float(pos.unrealized_plpc or 0.0)
            pnl_pct = plpc * 100.0
        except Exception:
            pnl_pct = 0.0

        should_sell = False
        if pnl_pct >= tp_pct:
            should_sell = True
            note = f"TP hit: +{pnl_pct:.2f}% ≥ {tp_pct:.2f}%"
        elif pnl_pct <= -sl_pct:
            should_sell = True
            note = f"SL hit: {pnl_pct:.2f}% ≤ -{sl_pct:.2f}%"
        else:
            note = f"Holding {symbol}: PnL {pnl_pct:.2f}% (TP {tp_pct:.2f}%, SL -{sl_pct:.2f}%)."

        if should_sell:
            try:
                qty = float(pos.qty)
            except Exception:
                qty = None  # unlikely; fallback to notional if needed
            try:
                action = "SELL"
                if qty and qty > 0:
                    place_market(OrderSide.SELL, qty=qty)
                else:
                    # rare fallback: sell by notional close to market value
                    notional_guess = abs(float(pos.market_value or 0.0))
                    if notional_guess <= 0 and mid_price:
                        notional_guess = mid_price * abs(float(pos.qty or 0.0))
                    if notional_guess <= 0:
                        raise ValueError("Cannot infer sell size")
                    place_market(OrderSide.SELL, notional_usd=notional_guess)
                note = f"{note} → Exited {symbol}."
            except Exception as e:
                action = "ERROR"
                note = f"SELL failed: {e}"
                print(f"ERROR: {note}")

    # Summary
    summary = {
        "symbol": symbol,
        "mid_price": mid_price,
        "position_present": bool(pos),
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "buy_usd": buy_usd,
        "dry_run": dry_run,
        "action": action,
        "order_id": order_id,
        "note": note,
        "ts": int(time.time()),
    }
    print("\n=== Alpaca Crypto Paper — Summary ===")
    print(json.dumps(summary, indent=2, sort_keys=True))

    write_json(STATE_DIR / "run_summary.json", summary)
    return 0 if action in ("HOLD", "BUY", "SELL") else 1

if __name__ == "__main__":
    sys.exit(main())
