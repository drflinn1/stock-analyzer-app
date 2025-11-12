#!/usr/bin/env python3
"""
alpaca_crypto_rotation.py
Alpaca PAPER rotation bot: pick the strongest from a universe (default BTC/ETH/SOL)
and hold only that one. On each run:

1) Score each symbol by ~24h performance (last 25 hourly bars).
2) Identify WINNER = highest 24h % change (fallback to 0% if bars unavailable).
3) If no crypto position: BUY WINNER (notional = BUY_USD).
4) If holding:
   - If holding WINNER: check TP/SL using unrealized_plpc; SELL if hit.
   - If holding NON-WINNER and ROTATE=ON: SELL current, then BUY WINNER.

Writes .state/run_summary.json and .state/rotation_scores.csv

Env:
  ALPACA_API_KEY, ALPACA_API_SECRET   (required)
  UNIVERSE       default "BTC/USD,ETH/USD,SOL/USD"
  BUY_USD        default "25"
  TP_PCT         default "5"     (sell if >= +TP_PCT)
  SL_PCT         default "2"     (sell if <= -SL_PCT)
  ROTATE         default "ON"    (ON to switch to new winner if different)
  DRY_RUN        default "ON"    (ON simulate only; OFF places PAPER orders)
  MAX_RETRY      default "3"
"""

from __future__ import annotations
import csv, json, math, os, sys, time
from pathlib import Path
from typing import List, Dict, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.common import APIError

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

STATE_DIR = Path(".state")
STATE_DIR.mkdir(exist_ok=True)

def getenv(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        return "" if default is None else str(default)
    s = str(v).strip()
    return s if s else ("" if default is None else str(default))

def to_float(s: str, fallback: float) -> float:
    try:
        return float(str(s).strip())
    except Exception:
        return fallback

def score_universe(data_cli: CryptoHistoricalDataClient, symbols: List[str]) -> Tuple[str, Dict[str, float], Dict]:
    """
    Score by ~24h change using last 25 hourly bars.
    Returns (winner, scores_dict, debug_dict)
    """
    scores: Dict[str, float] = {}
    debug: Dict[str, dict] = {}
    best_sym = symbols[0] if symbols else ""
    best = -math.inf

    for sym in symbols:
        chg = 0.0
        info = {"bars": 0, "close_now": None, "close_24h": None, "pct_24h": 0.0}
        try:
            req = CryptoBarsRequest(symbol_or_symbols=[sym], timeframe=TimeFrame.Hour, limit=25)
            bars_map = data_cli.get_crypto_bars(req)
            bars = bars_map.get(sym, [])
        except Exception as e:
            bars = []
            info["error"] = f"bars failed: {e}"

        if len(bars) >= 2:
            c0 = float(bars[0].close)
            c1 = float(bars[-1].close)
            info.update({"bars": len(bars), "close_24h": c0, "close_now": c1})
            chg = ((c1 - c0) / c0) * 100.0 if c0 > 0 else 0.0
        else:
            # Fallback to latest quote mid (treat change as 0 so it won't win unless all tie)
            try:
                q = CryptoLatestQuoteRequest(symbol_or_symbols=[sym])
                qd = data_cli.get_crypto_latest_quote(q).get(sym)
                if qd:
                    bid = float(qd.bid_price or 0.0)
                    ask = float(qd.ask_price or 0.0)
                    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else float(qd.midpoint or 0.0)
                    info["close_now"] = mid
            except Exception:
                pass

        info["pct_24h"] = chg
        scores[sym] = chg
        debug[sym] = info

        if chg > best:
            best = chg
            best_sym = sym

    debug["winner"] = best_sym
    return best_sym, scores, debug

def write_scores_csv(path: Path, scores: Dict[str, float]):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "pct_24h"])
        for k, v in scores.items():
            w.writerow([k, f"{v:.6f}"])

def main() -> int:
    api_key = getenv("ALPACA_API_KEY")
    api_secret = getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        print("ERROR: Missing ALPACA_API_KEY or ALPACA_API_SECRET")
        return 1

    universe = [s.strip() for s in getenv("UNIVERSE", "BTC/USD,ETH/USD,SOL/USD").split(",") if s.strip()]
    buy_usd  = to_float(getenv("BUY_USD", "25"), 25.0)
    tp_pct   = to_float(getenv("TP_PCT", "5"), 5.0)
    sl_pct   = to_float(getenv("SL_PCT", "2"), 2.0)
    rotate   = getenv("ROTATE", "ON").upper() == "ON"
    dry_run  = getenv("DRY_RUN", "ON").upper() == "ON"
    max_retry = int(to_float(getenv("MAX_RETRY", "3"), 3))

    trading = TradingClient(api_key, api_secret, paper=True)
    data_cli = CryptoHistoricalDataClient()

    winner, scores, debug = score_universe(data_cli, universe)
    write_scores_csv(STATE_DIR / "rotation_scores.csv", scores)

    # Find any current crypto position
    pos = None
    all_positions = []
    try:
        all_positions = trading.get_all_positions()
    except APIError as e:
        print(f"ERROR: fetching positions: {e}")
        return 1

    for p in all_positions:
        if "/" in p.symbol:  # Alpaca crypto symbols have a slash
            pos = p
            break

    action = "HOLD"
    order_id = None
    note = ""

    def market(side: OrderSide, symbol: str, *, notional: float | None = None, qty: float | None = None):
        nonlocal order_id
        if dry_run:
            print(f"DRY_RUN → would {side.name} {symbol} "
                  f"{'(notional=$%.2f)'%notional if notional else ''}"
                  f"{'(qty=%s)'%qty if qty else ''}")
            return
        req_kwargs = dict(symbol=symbol, time_in_force=TimeInForce.GTC)
        if notional is not None:
            req_kwargs["notional"] = float(notional)
        elif qty is not None:
            req_kwargs["qty"] = float(qty)
        else:
            raise ValueError("notional or qty required")
        o = trading.submit_order(order_data=MarketOrderRequest(side=side, **req_kwargs))
        order_id = str(o.id) if getattr(o, "id", None) is not None else None

    # Rotation logic
    if pos is None:
        # Flat → buy winner
        try:
            action = "BUY"
            market(OrderSide.BUY, winner, notional=buy_usd)
            note = f"Flat → bought winner {winner} with ${buy_usd:.2f}."
        except Exception as e:
            action = "ERROR"
            note = f"BUY failed: {e}"
    else:
        held = pos.symbol
        # Compute current PnL%
        try:
            pnl_pct = float(pos.unrealized_plpc or 0.0) * 100.0
        except Exception:
            pnl_pct = 0.0

        if held == winner:
            # manage TP/SL on winner
            if pnl_pct >= tp_pct:
                try:
                    q = float(pos.qty)
                    action = "SELL"
                    market(OrderSide.SELL, held, qty=q if q > 0 else None)
                    note = f"TP hit on {held}: +{pnl_pct:.2f}% ≥ {tp_pct:.2f}% → sold."
                except Exception as e:
                    action = "ERROR"
                    note = f"SELL failed (TP) on {held}: {e}"
            elif pnl_pct <= -sl_pct:
                try:
                    q = float(pos.qty)
                    action = "SELL"
                    market(OrderSide.SELL, held, qty=q if q > 0 else None)
                    note = f"SL hit on {held}: {pnl_pct:.2f}% ≤ -{sl_pct:.2f}% → sold."
                except Exception as e:
                    action = "ERROR"
                    note = f"SELL failed (SL) on {held}: {e}"
            else:
                action = "HOLD"
                note = f"Holding winner {held}. PnL {pnl_pct:.2f}% (TP {tp_pct:.2f}%, SL -{sl_pct:.2f}%)."
        else:
            # Holding non-winner
            if rotate:
                try:
                    q = float(pos.qty)
                    market(OrderSide.SELL, held, qty=q if q > 0 else None)
                    market(OrderSide.BUY, winner, notional=buy_usd)
                    action = "ROTATE"
                    note = f"Rotated {held} → {winner} (${buy_usd:.2f})."
                except Exception as e:
                    action = "ERROR"
                    note = f"ROTATE failed: {e}"
            else:
                action = "HOLD"
                note = f"Rotate=OFF. Still holding {held} while {winner} is winner."

    # Build summary
    summary = {
        "universe": universe,
        "winner": winner,
        "scores_pct": scores,
        "held_symbol": (pos.symbol if pos else None),
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "buy_usd": buy_usd,
        "rotate": rotate,
        "dry_run": dry_run,
        "action": action,
        "order_id": order_id,
        "note": note,
        "ts": int(time.time()),
    }

    print("\n=== Alpaca Crypto Rotation — Summary ===")
    print(json.dumps(summary, indent=2, sort_keys=True))

    (STATE_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if action in ("HOLD", "BUY", "SELL", "ROTATE") else 1

if __name__ == "__main__":
    sys.exit(main())
