#!/usr/bin/env python3
"""
tools/hourly_rotation_sim.py

Hourly 1-Coin Rotation simulator with:
- MIN_BUY_USD position sizing (default $25).
- Immediate re-rank and re-entry after any SELL (no delay).
- Optional risk-off: if BTC 60m return <= threshold, stay in cash.

Rules:
- Hold exactly 1 position at a time.
- Exit if: -1% from entry, or +5% from entry, or after 60 minutes if < +3%.
- Immediately rotate into current top gainer (risk-on only).

Data:
- Looks for minute/5m OHLCV CSVs in ./data/<SYMBOL>.csv with columns:
  timestamp,open,high,low,close,volume  (UTC timestamps).
- If no CSVs exist, it synthesizes toy data so you can see behavior.

Usage:
  python tools/hourly_rotation_sim.py --symbols BTC,ETH,SOL --bar_minutes 1 --fee_bps 10 --min_buy_usd 25 --risk_on 1 --risk_thresh_btc_60m -0.005
"""

from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path("data")
STATE_DIR = Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = STATE_DIR / "hourly_rotation_sim_log.csv"

# --- Strategy params (defaults can be overridden via CLI) ---
HOLD_WINDOW_MIN = 60
STOP_PCT = -0.01    # -1%
TP_PCT = 0.05       # +5%
MIN_1H_PCT = 0.03   # +3% @60m target
DEFAULT_FEE_BPS = 10  # 0.10% per side

@dataclass
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

def parse_ts(x: str) -> datetime:
    x = x.strip()
    if x.isdigit():
        return datetime.fromtimestamp(int(x), tz=timezone.utc)
    try:
        return datetime.fromisoformat(x.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return datetime.fromisoformat(x).replace(tzinfo=timezone.utc)

def load_csv(symbol: str, bar_minutes: int) -> List[Bar]:
    fp = DATA_DIR / f"{symbol}.csv"
    bars: List[Bar] = []
    if not fp.exists():
        # Synthesize toy data to demonstrate mechanics
        import random
        start = datetime.now(tz=timezone.utc) - timedelta(hours=24)
        price = 100.0 + random.random() * 10
        for i in range(24 * 60 // bar_minutes):
            ts = start + timedelta(minutes=i * bar_minutes)
            drift = (random.random() - 0.5) * 0.002
            spike = 0.0
            if random.random() < 0.02:
                spike = (random.random() - 0.5) * 0.1  # ±10% spike
            ret = drift + spike
            new_price = max(0.5, price * (1.0 + ret))
            o = price
            c = new_price
            h = max(o, c) * (1 + abs(ret) * 0.2)
            l = min(o, c) * (1 - abs(ret) * 0.2)
            bars.append(Bar(ts, o, h, l, c, volume=1000.0))
            price = new_price
        return bars

    with fp.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bars.append(Bar(
                ts=parse_ts(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0)),
            ))
    bars.sort(key=lambda b: b.ts)
    return bars

def pct_change(a: float, b: float) -> float:
    return 0.0 if a <= 0 else (b / a) - 1.0

def rank_universe(universe: Dict[str, List[Bar]], now_idx: Dict[str, int], lookback_minutes: int, bar_minutes: int) -> Optional[str]:
    # Pick symbol with highest return over the last 'lookback_minutes'
    best_sym, best_ret = None, -1e9
    lookback_bars = max(1, lookback_minutes // bar_minutes)
    for sym, bars in universe.items():
        i = now_idx[sym]
        if i <= lookback_bars:
            continue
        a = bars[i - lookback_bars].close
        b = bars[i].close
        r = pct_change(a, b)
        if r > best_ret:
            best_ret = r
            best_sym = sym
    return best_sym

def btc_risk_on(universe: Dict[str, List[Bar]], now_idx: Dict[str, int], bar_minutes: int, thresh_60m: float) -> bool:
    # If BTC exists, use its 60m return vs threshold; otherwise default True
    if "BTC" not in universe:
        return True
    lookback_bars = max(1, 60 // bar_minutes)
    i = now_idx["BTC"]
    if i <= lookback_bars:
        return True
    a = universe["BTC"][i - lookback_bars].close
    b = universe["BTC"][i].close
    r = pct_change(a, b)
    return r > thresh_60m

@dataclass
class Position:
    symbol: str
    entry_ts: datetime
    entry_px: float
    size_usd: float

def simulate(symbols: List[str], bar_minutes: int, fee_bps: int, min_buy_usd: float,
             use_risk: bool, risk_thresh_btc_60m: float) -> None:
    # Load bars
    universe = {sym: load_csv(sym, bar_minutes) for sym in symbols}
    now_idx = {sym: 0 for sym in symbols}
    max_len = max(len(bars) for bars in universe.values())

    cash = 10_000.0
    pos: Optional[Position] = None
    trades = 0
    wins = 0
    fees_paid = 0.0

    with LOG_PATH.open("w", newline="", encoding="utf-8") as lf:
        lw = csv.writer(lf)
        lw.writerow(["ts","event","symbol","price","cash","pos_symbol","pos_entry","unrealized_pct","note"])

        def log(ts, event, symbol, price, note=""):
            chg = 0.0
            if pos:
                chg = pct_change(pos.entry_px, price)
            lw.writerow([ts.isoformat(), event, symbol, f"{price:.6f}", f"{cash:.2f}",
                         pos.symbol if pos else "", f"{pos.entry_px:.6f}" if pos else "",
                         f"{chg:.4%}", note])

        # Helper: attempt immediate buy (re-rank) at current time/indices
        def try_immediate_buy(current_ts: datetime):
            nonlocal cash, pos, trades, fees_paid
            if cash < min_buy_usd:
                return  # not enough cash for MIN_BUY_USD
            if use_risk and not btc_risk_on(universe, now_idx, bar_minutes, risk_thresh_btc_60m):
                return  # risk-off: hold cash
            top = rank_universe(universe, now_idx, lookback_minutes=60, bar_minutes=bar_minutes)
            if top is None:
                return
            px = universe[top][now_idx[top]].close
            size_usd = min(cash, min_buy_usd)  # fixed ticket size
            fee = size_usd * (fee_bps / 10_000)
            fees_paid += fee
            size_usd_net = size_usd - fee
            pos = Position(symbol=top, entry_ts=current_ts, entry_px=px, size_usd=size_usd_net)
            cash -= size_usd  # reserve the ticket cost
            trades += 1
            log(current_ts, "BUY", top, px, note=f"size=${size_usd:.2f}, fee=${fee:.2f}")

        # Main loop
        for _ in range(max_len):
            # Determine current timestamp across symbols (max aligned)
            current_ts = None
            for sym, bars in universe.items():
                if now_idx[sym] < len(bars):
                    ts = bars[now_idx[sym]].ts
                    current_ts = ts if current_ts is None else max(current_ts, ts)

            if current_ts is None:
                break

            # Advance all symbols up to current_ts
            for sym, bars in universe.items():
                while now_idx[sym] + 1 < len(bars) and bars[now_idx[sym] + 1].ts <= current_ts:
                    now_idx[sym] += 1

            # Current prices
            prices = {sym: universe[sym][now_idx[sym]].close for sym in symbols}

            # If flat: try to buy immediately (risk-aware)
            if pos is None:
                try_immediate_buy(current_ts)
                continue

            # Manage open position
            px_now = prices.get(pos.symbol)
            if px_now is None:
                continue

            chg = pct_change(pos.entry_px, px_now)
            held_minutes = max(0, int((current_ts - pos.entry_ts).total_seconds() // 60))
            exit_reason = None

            if chg <= STOP_PCT:
                exit_reason = "STOP_-1%"
            elif chg >= TP_PCT:
                exit_reason = "TP_+5%"
            elif held_minutes >= HOLD_WINDOW_MIN and chg < MIN_1H_PCT:
                exit_reason = "FAIL_<+3%_@60m"

            if exit_reason:
                # Sell proceeds and fee
                proceeds = pos.size_usd * (1 + chg)
                fee = proceeds * (fee_bps / 10_000)
                fees_paid += fee
                cash += proceeds - fee
                wins += 1 if chg > 0 else 0
                log(current_ts, "SELL", pos.symbol, px_now, note=f"{exit_reason}, chg={chg:.2%}, fee=${fee:.2f}")
                pos = None

                # **Immediate re-rank and re-entry** at the same bar if risk-on
                try_immediate_buy(current_ts)
                continue

            # Heartbeat every 10 minutes
            if held_minutes % 10 == 0:
                log(current_ts, "HOLD", pos.symbol, px_now, note=f"chg={chg:.2%}, held={held_minutes}m")

    # End-of-sim liquidation if still holding
    if pos:
        last_px = universe[pos.symbol][now_idx[pos.symbol]].close
        chg = pct_change(pos.entry_px, last_px)
        proceeds = pos.size_usd * (1 + chg)
        fee = proceeds * (fee_bps / 10_000)
        fees_paid += fee
        cash += proceeds - fee
        wins += 1 if chg > 0 else 0

    # Summary
    start_equity = 10_000.0
    end_equity = cash
    pnl = end_equity - start_equity
    total_trades = trades
    win_rate = (wins / total_trades) if total_trades else 0.0

    print("=== Hourly 1-Coin Rotation (Sim) ===")
    print(f"Symbols: {','.join(symbols)}  |  Bar: {bar_minutes}m  |  Fees/side: {fee_bps/100:.2f}%")
    print(f"MIN_BUY_USD: ${min_buy_usd:.2f}  |  Risk filter: {'ON' if use_risk else 'OFF'} (BTC60m>{risk_thresh_btc_60m:.3%})")
    print(f"Trades: {total_trades}  |  Wins: {wins}  |  Win rate: {win_rate:.1%}")
    print(f"Fees paid: ${fees_paid:,.2f}")
    print(f"Start equity: ${start_equity:,.2f}  →  End equity: ${end_equity:,.2f}  (PnL: ${pnl:,.2f})")
    print(f"Run log: {LOG_PATH}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="BTC,ETH,SOL")
    ap.add_argument("--bar_minutes", type=int, default=1, help="1 or 5 are typical")
    ap.add_argument("--fee_bps", type=int, default=DEFAULT_FEE_BPS, help="per side; 10 bps = 0.10%")
    ap.add_argument("--min_buy_usd", type=float, default=25.0, help="ticket size per entry (USD)")
    ap.add_argument("--risk_on", type=int, default=1, help="1=enable risk filter, 0=disable")
    ap.add_argument("--risk_thresh_btc_60m", type=float, default=-0.005, help="BTC 60m return must be > threshold (e.g., -0.005 = -0.5%)")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    simulate(
        symbols=symbols,
        bar_minutes=args.bar_minutes,
        fee_bps=args.fee_bps,
        min_buy_usd=args.min_buy_usd,
        use_risk=bool(args.risk_on),
        risk_thresh_btc_60m=args.risk_thresh_btc_60m
    )

if __name__ == "__main__":
    main()
