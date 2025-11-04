#!/usr/bin/env python3
"""
Main unified runner — Hourly 1-Coin Rotation (with Auto-Universe)

New:
- AUTO_UNIVERSE=1 discovers Kraken USD spot pairs automatically, filters by USD volume,
  ranks by 60-minute return, and trades the top symbol (one position only).
- Manual UNIVERSE still supported and used as a fallback/override.

Exits (in priority):
 1) STOP_-1%
 2) TP_+5%
 3) FAIL_<+3%@60m  (after 60 minutes since entry, if gain < +3%)

After any SELL: immediate re-rank and re-enter (risk-aware).
"""

from __future__ import annotations
import csv
import json
import math
import os
import re
import sys
import ssl
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -------------------- Env helpers --------------------
def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else str(v).strip()

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

# -------------------- Config --------------------
STATE_DIR = Path(env_str("STATE_DIR", ".state"))
STATE_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_JSON = STATE_DIR / "run_summary.json"
SUMMARY_MD   = STATE_DIR / "run_summary.md"
POSITION_JSON= STATE_DIR / "position.json"
RUN_LOG_CSV  = STATE_DIR / "hourly_rotation_runlog.csv"
CANDS_CSV    = STATE_DIR / "momentum_candidates.csv"  # optional manual input

# Strategy vars
DRY_RUN  = env_str("DRY_RUN", "ON").upper() != "OFF"
QUOTE    = env_str("QUOTE", "USD").upper()
UNIVERSE = [s.strip().upper() for s in env_str("UNIVERSE", "").split(",") if s.strip()]

# Auto-universe controls
AUTO_UNIVERSE          = env_int("AUTO_UNIVERSE", 1) == 1     # 1=on, 0=off
AUTO_UNIVERSE_TOP_K    = env_int("AUTO_UNIVERSE_TOP_K", 30)   # rank up to this many by 60m return
AUTO_MIN_BASE_VOL_USD  = env_float("AUTO_MIN_BASE_VOL_USD", 25000.0)
AUTO_EXCLUDE           = {s.strip().upper() for s in env_str("AUTO_EXCLUDE", "USDT,USDC,EUR,DAI,FDUSD").split(",") if s.strip()}  # symbols to avoid

# Tickets / fees
MIN_BUY_USD = env_float("MIN_BUY_USD", 25.0)
FEE_BPS     = env_int("FEE_BPS", 10)  # per side; 10 bps = 0.10%

# Exit rules
HOLD_WINDOW_MIN = 60
STOP_PCT   = -0.01
TP_PCT     = 0.05
MIN_1H_PCT = 0.03

# Risk gate
RISK_ON = env_int("RISK_ON", 1) == 1
RISK_THRESH_BTC_60M = env_float("RISK_THRESH_BTC_60M", -0.005)  # > -0.5% to be risk-on

# -------------------- Types --------------------
@dataclass
class Position:
    symbol: str
    entry_ts: str
    entry_px: float
    size_usd: float
    quote: str = QUOTE

    @property
    def dt(self) -> datetime:
        return datetime.fromisoformat(self.entry_ts)

# -------------------- Kraken public API (no auth) --------------------
_SSL = ssl.create_default_context()

def http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=_SSL, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def kr_pair(symbol: str, quote: str) -> str:
    s = symbol.upper()
    if s == "BTC": s = "XBT"
    return f"{s}{quote.upper()}"

def get_last_price(symbol: str, quote: str = QUOTE) -> Optional[float]:
    pair = kr_pair(symbol, quote)
    url = f"https://api.kraken.com/0/public/Ticker?pair={urllib.parse.quote(pair)}"
    try:
        data = http_json(url)
        if data.get("error"):
            return None
        r = next(iter(data["result"].values()))
        return float(r["c"][0])
    except Exception:
        return None

def get_ret_60m(symbol: str, quote: str = QUOTE, interval: int = 1) -> Optional[float]:
    pair = kr_pair(symbol, quote)
    url = f"https://api.kraken.com/0/public/OHLC?pair={urllib.parse.quote(pair)}&interval={interval}"
    try:
        data = http_json(url)
        if data.get("error"):
            return None
        rows = next(iter(data["result"].values()))
        if len(rows) < 65:
            return None
        c_now  = float(rows[-1][4])
        c_60m  = float(rows[-61][4])
        if c_60m <= 0: return None
        return (c_now / c_60m) - 1.0
    except Exception:
        return None

def discover_usd_spot_pairs() -> List[str]:
    """
    Returns Kraken symbols (BTC, ETH, SOL, ...) whose pair key ends with 'USD'.
    """
    url = "https://api.kraken.com/0/public/AssetPairs"
    try:
        data = http_json(url)
        if data.get("error"):
            return []
        syms: List[str] = []
        for pair_key, meta in data["result"].items():
            # We only want spot USD pairs like XBTUSD, ETHUSD
            if not pair_key.endswith("USD"):
                continue
            base = pair_key[:-3]  # strip USD
            if base == "XBT":
                base = "BTC"
            # skip excluded bases (stablecoins, fiat proxies)
            if base.upper() in AUTO_EXCLUDE:
                continue
            syms.append(base.upper())
        return sorted(list(set(syms)))
    except Exception:
        return []

def ticker_24h(symbols: List[str], quote: str = QUOTE) -> Dict[str, Dict[str, float]]:
    """
    Fetch last price and 24h base volume for many symbols at once.
    Returns {SYM: {"last": float, "base_vol": float, "usd_vol": float}}
    """
    results: Dict[str, Dict[str, float]] = {}
    if not symbols:
        return results
    # Build Kraken pair list
    pairs = [kr_pair(s, quote) for s in symbols]
    url = "https://api.kraken.com/0/public/Ticker?pair=" + urllib.parse.quote(",".join(pairs))
    try:
        data = http_json(url)
        if data.get("error"):
            return results
        # Kraken result keys may alias; map back by order
        # We'll build a reverse index by symbol to be safe.
        for k, v in data["result"].items():
            # Determine base symbol from pair key
            base = k[:-3]  # drop USD
            if base == "XBT": base = "BTC"
            sym = base.upper()
            last = float(v["c"][0])
            base_vol = float(v["v"][1])  # 24h volume (base units)
            usd_vol = base_vol * last
            results[sym] = {"last": last, "base_vol": base_vol, "usd_vol": usd_vol}
        return results
    except Exception:
        return results

# -------------------- Universe & ranking --------------------
def load_candidates_csv(path: Path) -> List[str]:
    if not path.exists():
        return []
    out: List[Tuple[float, str]] = []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                sym = row.get("symbol", "").strip().upper()
                rk  = row.get("rank", "")
                try:
                    rk = float(rk)
                except Exception:
                    rk = math.inf
                if sym:
                    out.append((rk, sym))
        out.sort(key=lambda x: x[0])
        return [s for _, s in out]
    except Exception:
        return []

def build_universe() -> List[str]:
    # Manual override
    if UNIVERSE:
        return list(dict.fromkeys(UNIVERSE))

    # Auto discovery
    if AUTO_UNIVERSE:
        bases = discover_usd_spot_pairs()
        if not bases:
            return []
        t = ticker_24h(bases, quote=QUOTE)
        # filter by USD volume
        filt = [s for s in bases if t.get(s, {}).get("usd_vol", 0.0) >= AUTO_MIN_BASE_VOL_USD]
        # Keep a sane cap; we will rank later by 60m return
        return sorted(list(dict.fromkeys(filt)))[: max(1, AUTO_UNIVERSE_TOP_K)]

    # Fallback to CSV candidates
    cands = load_candidates_csv(CANDS_CSV)
    return cands[: max(1, AUTO_UNIVERSE_TOP_K)] if cands else []

def rank_top_symbol(candidates: List[str]) -> Optional[str]:
    # Rank by 60m return; if ties/missing, fallback to CSV order, then first
    best_sym, best_ret = None, -1e9
    for s in candidates:
        r = get_ret_60m(s, quote=QUOTE, interval=1)
        if r is None:
            continue
        if r > best_ret:
            best_ret, best_sym = r, s
    if best_sym:
        return best_sym

    # Fallbacks
    csv_syms = load_candidates_csv(CANDS_CSV)
    for s in csv_syms:
        if s in candidates:
            return s
    return candidates[0] if candidates else None

# -------------------- Risk gate --------------------
def btc_risk_ok() -> bool:
    if not RISK_ON:
        return True
    r = get_ret_60m("BTC", quote=QUOTE, interval=1)
    if r is None:
        return True  # don't freeze if API hiccups
    return r > RISK_THRESH_BTC_60M

# -------------------- Broker (DRY-RUN safe) --------------------
class Broker:
    def __init__(self, dry_run: bool, fee_bps: int):
        self.dry_run = dry_run
        self.fee_bps = fee_bps

    def place_market(self, side: str, symbol: str, usd_amount: float, price: Optional[float] = None) -> Dict:
        """
        side: 'buy' or 'sell'; usd_amount is ticket estimate
        """
        if self.dry_run:
            return {"status": "ok", "dry_run": True, "side": side, "symbol": symbol, "usd_amount": usd_amount}
        # ---------- LIVE HOOK ----------
        # Replace this with your Kraken adapter call when ready.
        print(f"[LIVE INTENT] {side.upper()} {symbol}/{QUOTE} ~ ${usd_amount:.2f} @ ~{price or 0:.6f}")
        return {"status": "ok", "dry_run": False, "side": side, "symbol": symbol, "usd_amount": usd_amount}

# -------------------- Position state --------------------
def read_pos() -> Optional[Position]:
    if not POSITION_JSON.exists():
        return None
    try:
        return Position(**json.loads(POSITION_JSON.read_text()))
    except Exception:
        return None

def write_pos(p: Optional[Position]) -> None:
    if p is None:
        if POSITION_JSON.exists():
            POSITION_JSON.unlink()
        return
    POSITION_JSON.write_text(json.dumps(asdict(p), indent=2))

# -------------------- Logging --------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def pct_change(a: float, b: float) -> float:
    if a <= 0: return 0.0
    return (b / a) - 1.0

def append_runlog(ts: datetime, event: str, symbol: str, price: float, note: str, pos: Optional[Position]):
    newfile = not RUN_LOG_CSV.exists()
    with RUN_LOG_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(["ts","event","symbol","price","pos_symbol","entry_px","unrealized_pct","note"])
        chg = ""
        if pos and price:
            chg = f"{pct_change(pos.entry_px, price):.4%}"
        w.writerow([ts.isoformat(), event, symbol, f"{price:.6f}", pos.symbol if pos else "",
                    f"{pos.entry_px if pos else 0:.6f}", chg, note])

def write_summary(data: Dict):
    SUMMARY_JSON.write_text(json.dumps(data, indent=2))
    SUMMARY_MD.write_text("\n".join([
        f"**When:** {data.get('when')}",
        f"**DRY_RUN:** {'ON' if DRY_RUN else 'OFF'}",
        f"**Action:** {data.get('action')}",
        f"**Symbol:** {data.get('symbol','')}",
        f"**Note:** {data.get('note','')}"
    ]))

# -------------------- Core rotation (single pass per run) --------------------
def main() -> int:
    ts = now_utc()
    broker = Broker(DRY_RUN, FEE_BPS)
    pos = read_pos()

    # Flat → enter
    if pos is None:
        if not btc_risk_ok():
            note = "Risk-OFF → hold cash"
            write_summary({"when": ts.isoformat(), "action": "HOLD_CASH", "note": note})
            append_runlog(ts, "HOLD_CASH", "", 0.0, note, None)
            print(note)
            return 0

        candidates = build_universe()
        if not candidates:
            # fallback to CSV or first-listed big caps
            fallback = load_candidates_csv(CANDS_CSV) or ["BTC","ETH","SOL"]
            candidates = fallback

        top = rank_top_symbol(candidates)
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

        fee = MIN_BUY_USD * (FEE_BPS / 10_000.0)
        usd_net = max(0.0, MIN_BUY_USD - fee)

        broker.place_market("buy", top, MIN_BUY_USD, price=px)
        new_pos = Position(symbol=top, entry_ts=ts.isoformat(), entry_px=px, size_usd=usd_net, quote=QUOTE)
        write_pos(new_pos)
        append_runlog(ts, "BUY", top, px, f"size=${MIN_BUY_USD:.2f}, fee=${fee:.2f}", new_pos)
        write_summary({"when": ts.isoformat(), "action": "BUY", "symbol": top, "note": f"size ${MIN_BUY_USD:.2f}"})
        print(f"BUY {top}/{QUOTE} ~{px:.6f} | ticket ${MIN_BUY_USD:.2f}")
        return 0

    # Holding → check exits
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
        fee = proceeds * (FEE_BPS / 10_000.0)
        usd_after = max(0.0, proceeds - fee)
        broker.place_market("sell", pos.symbol, proceeds, price=px_now)
        append_runlog(ts, "SELL", pos.symbol, px_now, f"{exit_reason}, chg={chg:.2%}, fee=${fee:.2f}", pos)
        write_pos(None)

        # Immediate rotation (risk-aware)
        if btc_risk_ok():
            candidates = build_universe()
            if not candidates:
                candidates = ["BTC","ETH","SOL"]
            top = rank_top_symbol(candidates)
            if top:
                px = get_last_price(top, QUOTE)
                if px is not None:
                    ticket = MIN_BUY_USD
                    fee_b = ticket * (FEE_BPS / 10_000.0)
                    usd_net = max(0.0, ticket - fee_b)
                    broker.place_market("buy", top, ticket, price=px)
                    new_pos = Position(symbol=top, entry_ts=ts.isoformat(), entry_px=px, size_usd=usd_net, quote=QUOTE)
                    write_pos(new_pos)
                    append_runlog(ts, "BUY", top, px, f"rotate; size=${ticket:.2f}, fee=${fee_b:.2f}", new_pos)
                    write_summary({"when": ts.isoformat(), "action": "ROTATE", "symbol": f"{pos.symbol}→{top}", "note": exit_reason})
                    print(f"ROTATE {pos.symbol}→{top} @ ~{px:.6f}")
                    return 0

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
