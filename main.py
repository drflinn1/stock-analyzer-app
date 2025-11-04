#!/usr/bin/env python3
"""
Main unified runner — Hourly 1-Coin Rotation

Rules
-----
- Hold exactly 1 coin.
- Exit if:
  1) chg <= -1% (hard stop)
  2) chg >= +5% (take profit)
  3) after 60 minutes since entry, chg < +3%  -> exit
- After any exit: immediately re-rank and re-enter the current top gainer.
- Optional risk-off: if BTC 60m return <= threshold, hold cash.

This file is DRY-RUN safe by default. Set DRY_RUN=OFF to enable live orders.
Live orders section is centralized in Broker.place_market() — replace that
with your existing Kraken adapter if you prefer.

Environment (sensible defaults)
-------------------------------
DRY_RUN=ON|OFF                (default ON)
UNIVERSE="BTC,ETH,SOL"        (symbols to scan; comma-separated)
MIN_BUY_USD=25                (ticket size per entry)
FEE_BPS=10                    (per side; 10 bps = 0.10%)
RISK_ON=1                     (1=enable risk filter, 0=disable)
RISK_THRESH_BTC_60M=-0.005    (BTC 60m return must be > threshold)
QUOTE=USD                     (quote currency)
STATE_DIR=.state              (folder for run artifacts)
"""

from __future__ import annotations
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
import urllib.request
import urllib.parse
import ssl

# ---------- Config / helpers ----------
STATE_DIR = Path(os.getenv("STATE_DIR", ".state"))
STATE_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_JSON = STATE_DIR / "run_summary.json"
SUMMARY_MD   = STATE_DIR / "run_summary.md"
POSITION_JSON= STATE_DIR / "position.json"
RUN_LOG_CSV  = STATE_DIR / "hourly_rotation_runlog.csv"
CANDS_CSV    = STATE_DIR / "momentum_candidates.csv"  # optional

def env_str(name: str, default: str) -> str:
    v = os.getenv(name, default)
    return default if v is None else str(v).strip()

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

# Strategy params
DRY_RUN = env_str("DRY_RUN", "ON").upper() != "OFF"
UNIVERSE = [s.strip().upper() for s in env_str("UNIVERSE", "BTC,ETH,SOL").split(",") if s.strip()]
QUOTE    = env_str("QUOTE", "USD").upper()

MIN_BUY_USD = env_float("MIN_BUY_USD", 25.0)
FEE_BPS     = env_int("FEE_BPS", 10)  # per side

HOLD_WINDOW_MIN = 60
STOP_PCT = -0.01
TP_PCT   = 0.05
MIN_1H_PCT = 0.03

RISK_ON  = env_int("RISK_ON", 1) == 1
RISK_THRESH_BTC_60M = env_float("RISK_THRESH_BTC_60M", -0.005)  # -0.5%

# ---------- Data types ----------
@dataclass
class Position:
    symbol: str
    entry_ts: str      # ISO8601
    entry_px: float
    size_usd: float    # after entry fee
    quote: str = QUOTE

    @property
    def dt(self) -> datetime:
        return datetime.fromisoformat(self.entry_ts)

# ---------- Kraken public OHLC / Ticker (no auth) ----------
_SSLCTX = ssl.create_default_context()

def _http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, context=_SSLCTX, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))

def _kraken_pair(symbol: str, quote: str) -> str:
    # Kraken uses XBT instead of BTC. Many spot pairs are like XBTUSD, ETHUSD, SOLUSD.
    s = symbol.upper()
    if s == "BTC": s = "XBT"
    return f"{s}{quote.upper()}"

def get_last_price(symbol: str, quote: str = QUOTE) -> Optional[float]:
    # Use public Ticker (faster than OHLC)
    pair = _kraken_pair(symbol, quote)
    url = f"https://api.kraken.com/0/public/Ticker?pair={urllib.parse.quote(pair)}"
    try:
        data = _http_json(url)
        if data.get("error"):
            return None
        # Result dict key can be renamed by Kraken; take first
        r = next(iter(data["result"].values()))
        last = float(r["c"][0])  # last trade price
        return last
    except Exception:
        return None

def get_return_60m(symbol: str, bar_minutes: int = 1, quote: str = QUOTE) -> Optional[float]:
    # Compute 60m return using OHLC
    pair = _kraken_pair(symbol, quote)
    interval = max(1, bar_minutes)
    url = f"https://api.kraken.com/0/public/OHLC?pair={urllib.parse.quote(pair)}&interval={interval}"
    try:
        data = _http_json(url)
        if data.get("error"):
            return None
        rows = next(iter(data["result"].values()))
        if len(rows) < 70:
            return None
        # last row close vs row 60 bars back (approx 60m if interval=1)
        close_now = float(rows[-1][4])
        close_prev= float(rows[-61][4]) if len(rows) > 61 else float(rows[0][4])
        if close_prev <= 0:
            return None
        return (close_now / close_prev) - 1.0
    except Exception:
        return None

# ---------- Ranking ----------
def load_candidates_csv(path: Path) -> List[str]:
    # Optional: if you already produce .state/momentum_candidates.csv
    # with columns: symbol,quote,rank — we honor rank
    if not path.exists():
        return []
    syms = []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            rows = []
            for row in r:
                sym = row.get("symbol","").strip().upper()
                rk  = row.get("rank","")
                try:
                    rk = float(rk)
                except Exception:
                    rk = math.inf
                if sym:
                    rows.append((rk, sym))
            rows.sort(key=lambda x: x[0])
            syms = [s for _, s in rows]
    except Exception:
        pass
    return syms

def rank_top_gainer(universe: List[str]) -> Optional[str]:
    # Try 60m returns from Kraken public OHLC
    rets: List[Tuple[float,str]] = []
    for s in universe:
        r = get_return_60m(s, bar_minutes=1, quote=QUOTE)
        if r is None:
            continue
        rets.append((r, s))
    if rets:
        rets.sort(reverse=True, key=lambda t: t[0])
        return rets[0][1]

    # Fallback: use CSV candidates (rank ascending)
    csv_syms = load_candidates_csv(CANDS_CSV)
    if csv_syms:
        return csv_syms[0]

    # Last fallback: first in UNIVERSE
    return universe[0] if universe else None

# ---------- Risk gate ----------
def risk_ok() -> bool:
    if not RISK_ON:
        return True
    r = get_return_60m("BTC", bar_minutes=1, quote=QUOTE)
    if r is None:
        # If we can’t compute, default to ON to avoid idling forever.
        return True
    return r > RISK_THRESH_BTC_60M

# ---------- Broker (DRY-RUN safe) ----------
class Broker:
    def __init__(self, dry_run: bool, fee_bps: int):
        self.dry_run = dry_run
        self.fee_bps = fee_bps

    def place_market(self, side: str, symbol: str, usd_amount: float, price: Optional[float]=None) -> Dict:
        """
        side: 'buy' or 'sell'
        usd_amount: ticket size in USD (for sell, we use position size_usd*(1+chg))
        """
        if self.dry_run:
            return {"status":"ok","dry_run":True,"side":side,"symbol":symbol,"usd_amount":usd_amount}
        # -------- LIVE ORDERS ----------
        # Replace with your existing Kraken adapter, e.g.:
        #   kraken.place_market(symbol, side, usd_amount, quote=QUOTE)
        # For safety, we only print an intent here.
        print(f"[LIVE INTENT] {side.upper()} {symbol}/{QUOTE} for ~${usd_amount:.2f}")
        return {"status":"ok","dry_run":False,"side":side,"symbol":symbol,"usd_amount":usd_amount}

# ---------- Position state ----------
def read_pos() -> Optional[Position]:
    if not POSITION_JSON.exists():
        return None
    try:
        d = json.loads(POSITION_JSON.read_text())
        return Position(**d)
    except Exception:
        return None

def write_pos(p: Optional[Position]) -> None:
    if p is None:
        if POSITION_JSON.exists():
            POSITION_JSON.unlink()
        return
    POSITION_JSON.write_text(json.dumps(asdict(p), indent=2))

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def pct_change(a: float, b: float) -> float:
    if a <= 0: return 0.0
    return (b/a) - 1.0

# ---------- Logging ----------
def append_runlog(ts: datetime, event: str, symbol: str, price: float, note: str, pos: Optional[Position]):
    newfile = not RUN_LOG_CSV.exists()
    with RUN_LOG_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(["ts","event","symbol","price","pos_symbol","entry_px","unrealized_pct","note"])
        chg = ""
        if pos and price:
            chg = f"{pct_change(pos.entry_px, price):.4%}"
        w.writerow([ts.isoformat(), event, symbol, f"{price:.6f}", pos.symbol if pos else "", f"{pos.entry_px if pos else 0:.6f}", chg, note])

def write_summary(data: Dict):
    SUMMARY_JSON.write_text(json.dumps(data, indent=2))
    lines = [
        f"**When:** {data.get('when')}",
        f"**DRY_RUN:** {'ON' if DRY_RUN else 'OFF'}",
        f"**Action:** {data.get('action')}",
        f"**Symbol:** {data.get('symbol','')}",
        f"**Note:** {data.get('note','')}",
    ]
    SUMMARY_MD.write_text("\n".join(lines))

# ---------- Core rotation loop (single pass per run) ----------
def main() -> int:
    broker = Broker(DRY_RUN, FEE_BPS)
    pos = read_pos()
    ts = now_utc()

    # If flat → try to buy the top gainer (respect risk gate)
    if pos is None:
        if not risk_ok():
            note = "Risk-OFF → hold cash"
            write_summary({"when": ts.isoformat(), "action": "HOLD_CASH", "note": note})
            append_runlog(ts, "HOLD_CASH", "", 0.0, note, None)
            print(note)
            return 0

        top = rank_top_gainer(UNIVERSE)
        if not top:
            note = "No candidate found"
            write_summary({"when": ts.isoformat(), "action": "NO_CANDIDATE", "note": note})
            append_runlog(ts, "NO_CANDIDATE", "", 0.0, note, None)
            print(note)
            return 0

        px = get_last_price(top, QUOTE)
        if px is None:
            note = f"No price for {top}/{QUOTE}"
            write_summary({"when": ts.isoformat(), "action": "NO_PRICE", "symbol": top, "note": note})
            append_runlog(ts, "NO_PRICE", top, 0.0, note, None)
            print(note)
            return 0

        fee = MIN_BUY_USD * (FEE_BPS/10_000.0)
        usd_net = max(0.0, MIN_BUY_USD - fee)

        broker.place_market("buy", top, MIN_BUY_USD, price=px)
        new_pos = Position(symbol=top, entry_ts=ts.isoformat(), entry_px=px, size_usd=usd_net, quote=QUOTE)
        write_pos(new_pos)
        append_runlog(ts, "BUY", top, px, f"size=${MIN_BUY_USD:.2f}, fee=${fee:.2f}", new_pos)
        write_summary({"when": ts.isoformat(), "action": "BUY", "symbol": top, "note": f"size ${MIN_BUY_USD:.2f}"})
        print(f"BUY {top}/{QUOTE} at ~{px:.6f} | ticket ${MIN_BUY_USD:.2f}")
        return 0

    # If holding → apply exits
    px_now = get_last_price(pos.symbol, pos.quote)
    if px_now is None:
        note = f"No price for {pos.symbol}/{pos.quote}; continue holding"
        write_summary({"when": ts.isoformat(), "action": "HOLD_NO_PRICE", "symbol": pos.symbol, "note": note})
        append_runlog(ts, "HOLD", pos.symbol, 0.0, note, pos)
        print(note)
        return 0

    chg = pct_change(pos.entry_px, px_now)
    held_min = max(0, int((ts - pos.dt).total_seconds() // 60))
    exit_reason = None
    if chg <= STOP_PCT:
        exit_reason = "STOP_-1%"
    elif chg >= TP_PCT:
        exit_reason = "TP_+5%"
    elif held_min >= HOLD_WINDOW_MIN and chg < MIN_1H_PCT:
        exit_reason = "FAIL_<+3%@60m"

    if exit_reason:
        proceeds = pos.size_usd * (1.0 + chg)
        fee = proceeds * (FEE_BPS/10_000.0)
        usd_after = max(0.0, proceeds - fee)
        broker.place_market("sell", pos.symbol, proceeds, price=px_now)
        append_runlog(ts, "SELL", pos.symbol, px_now, f"{exit_reason}, chg={chg:.2%}, fee=${fee:.2f}", pos)
        write_pos(None)  # flat

        # Immediate re-rank & re-enter (risk-aware)
        if risk_ok():
            top = rank_top_gainer(UNIVERSE)
            if top:
                px = get_last_price(top, QUOTE)
                if px is not None:
                    ticket = MIN_BUY_USD
                    fee_b = ticket * (FEE_BPS/10_000.0)
                    usd_net = max(0.0, ticket - fee_b)
                    broker.place_market("buy", top, ticket, price=px)
                    new_pos = Position(symbol=top, entry_ts=ts.isoformat(), entry_px=px, size_usd=usd_net, quote=QUOTE)
                    write_pos(new_pos)
                    append_runlog(ts, "BUY", top, px, f"rotate; size=${ticket:.2f}, fee=${fee_b:.2f}", new_pos)
                    write_summary({"when": ts.isoformat(), "action": "ROTATE", "symbol": f"{pos.symbol}→{top}", "note": exit_reason})
                    print(f"ROTATE {pos.symbol}→{top} at ~{px:.6f}")
                    return 0

        # Could not (or chose not to) re-enter
        write_summary({"when": ts.isoformat(), "action": "SELL_TO_CASH", "symbol": pos.symbol, "note": exit_reason})
        print(f"SELL→CASH {pos.symbol} | {exit_reason}")
        return 0

    # No exit → heartbeat
    append_runlog(ts, "HOLD", pos.symbol, px_now, f"chg={chg:.2%}, held={held_min}m", pos)
    write_summary({"when": ts.isoformat(), "action": "HOLD", "symbol": pos.symbol, "note": f"chg={chg:.2%}, held={held_min}m"})
    print(f"HOLD {pos.symbol} | chg={chg:.2%} held={held_min}m")
    return 0

if __name__ == "__main__":
    sys.exit(main())
