#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reset / Restore helper for CryptoBot.

Modes:
  - FORCE_LIQUIDATE: Sell all non-stable positions (USD/USDT quotes preferred),
                     skipping tiny notional positions below MIN_SELL_USD,
                     unless you pass --include-dust.
  - CLEAN_REINIT   : Delete .state files and (optionally) dust sweep.
  - SEPTEMBER_BASELINE: Wipe .state and write a baseline guard-pack config snapshot.

Environment defaults (can be overridden with CLI flags):
  DRY_RUN, MIN_SELL_USD, DUST_MIN_USD, DUST_SKIP_STABLES, UNIVERSE_TOP_K,
  MAX_POSITIONS, MAX_BUYS_PER_RUN, ROTATE_WHEN_FULL, ROTATE_WHEN_CASH_SHORT,
  RESERVE_CASH_PCT, KRAKEN_API_KEY, KRAKEN_API_SECRET

Outputs:
  .state/reset_report.json  — detailed summary of actions taken.

Requirements:
  - ccxt (for live Kraken ops). If missing or DRY_RUN=ON, the script simulates.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

# ---------- Safe ccxt import (optional) ----------
_CCXT_AVAILABLE = True
try:
    import ccxt  # type: ignore
except Exception:
    _CCXT_AVAILABLE = False

STATE_DIR = Path(".state")
REPORT_PATH = STATE_DIR / "reset_report.json"
POS_FILE = STATE_DIR / "positions.json"
KPI_FILE = STATE_DIR / "kpi_history.csv"
LOG_DIR = Path(".logs")  # optional, if your repo uses this

STABLE_KEYWORDS = {"USD", "USDT", "USDC", "DAI", "EUR", "GBP"}

# ---------- Utilities ----------
def env(key: str, default: Optional[str] = None) -> str:
    val = os.getenv(key)
    return val if val is not None and val != "" else (default if default is not None else "")

def to_bool(x: str | bool | None, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    x = x.strip().lower()
    return x in {"1", "true", "yes", "on"}

def is_stable_asset(ticker_or_asset: str) -> bool:
    # crude but effective: match common stable symbols or quotes
    t = ticker_or_asset.upper()
    return any(token in t for token in STABLE_KEYWORDS)

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

@dataclass
class SellAction:
    asset: str
    symbol: str
    amount: float
    price: float
    notional: float
    executed: bool
    reason: str

@dataclass
class DustAction:
    asset: str
    amount: float
    notional: float
    skipped: bool
    reason: str

@dataclass
class Report:
    started_at: str
    finished_at: Optional[str]
    mode: str
    dry_run: bool
    ccxt_available: bool
    summary: Dict[str, Any]
    sells: List[SellAction]
    dust: List[DustAction]
    files_deleted: List[str]
    files_written: List[str]
    notes: List[str]

    def write(self, path: Path = REPORT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "mode": self.mode,
                "dry_run": self.dry_run,
                "ccxt_available": self.ccxt_available,
                "summary": self.summary,
                "sells": [asdict(s) for s in self.sells],
                "dust": [asdict(d) for d in self.dust],
                "files_deleted": self.files_deleted,
                "files_written": self.files_written,
                "notes": self.notes,
            }, f, indent=2)


# ---------- Kraken / ccxt helpers ----------
class KrakenClient:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self.client = None
        if _CCXT_AVAILABLE and not dry_run:
            key = env("KRAKEN_API_KEY")
            secret = env("KRAKEN_API_SECRET")
            if not key or not secret:
                raise RuntimeError("Missing KRAKEN_API_KEY / KRAKEN_API_SECRET for live operations.")
            self.client = ccxt.kraken({
                "apiKey": key,
                "secret": secret,
                "enableRateLimit": True,
            })

    def markets(self) -> Dict[str, Any]:
        if self.client:
            return self.client.load_markets()
        return {}  # not used in dry-run

    def fetch_balance(self) -> Dict[str, Any]:
        if self.client:
            return self.client.fetch_balance()
        # Simulated minimal structure
        return {"total": {}}

    def price_for(self, base: str, prefer_quotes: List[str]) -> (str, float):
        """
        Return (symbol, last_price) for base against the first available quote in prefer_quotes.
        Tries exact "BASE/QUOTE" first, then scans markets.
        """
        base_u = base.upper()
        if self.client:
            markets = self.client.load_markets()
            for q in prefer_quotes:
                sym_try = f"{base_u}/{q}"
                if sym_try in markets:
                    ticker = self.client.fetch_ticker(sym_try)
                    return sym_try, float(ticker.get("last") or ticker.get("close") or 0.0)

            # fallback: first USD-like quote available
            for m, mdat in markets.items():
                if mdat.get("base", "").upper() == base_u and mdat.get("quote", "").upper() in STABLE_KEYWORDS:
                    ticker = self.client.fetch_ticker(m)
                    return m, float(ticker.get("last") or ticker.get("close") or 0.0)
            return "", 0.0
        # dry-run: pretend USD pair exists at fake price
        return f"{base_u}/USD", 1.0

    def market_sell(self, symbol: str, amount: float) -> Dict[str, Any]:
        if self.client and not self.dry_run:
            order = self.client.create_order(symbol=symbol, type="market", side="sell", amount=amount)
            return order
        # simulate
        return {"id": "SIM-SELL", "symbol": symbol, "amount": amount, "status": "filled"}


# ---------- Core actions ----------
def force_liquidate(
    kraken: KrakenClient,
    min_sell_usd: float,
    include_dust: bool,
    skip_stables: bool,
    report: Report,
) -> None:
    bal = kraken.fetch_balance()
    totals: Dict[str, float] = (bal.get("total") or {}) if isinstance(bal, dict) else {}
    prefer_quotes = ["USD", "USDT"]

    for asset, amount in sorted(totals.items()):
        try:
            amt = float(amount or 0.0)
        except Exception:
            continue
        if amt <= 0.0:
            continue

        # Skip known stables if requested
        if skip_stables and is_stable_asset(asset):
            report.notes.append(f"Skip stable holding: {asset}")
            continue

        symbol, price = kraken.price_for(asset, prefer_quotes)
        notional = amt * float(price or 0.0)

        # Decide: dust vs sell
        if not include_dust and notional < min_sell_usd:
            report.dust.append(DustAction(asset=asset, amount=amt, notional=notional, skipped=True,
                                          reason=f"Below MIN_SELL_USD={min_sell_usd}"))
            continue

        executed = False
        reason = ""
        if symbol and price > 0.0:
            try:
                kraken.market_sell(symbol, amt)
                executed = True
                reason = "Sold at market"
            except Exception as e:
                executed = False
                reason = f"Sell failed: {e}"
        else:
            reason = "No valid USD/USDT market or price"

        report.sells.append(SellAction(
            asset=asset, symbol=symbol or f"{asset}/USD?",
            amount=amt, price=price, notional=notional,
            executed=executed, reason=reason
        ))


def clean_reinit(
    dust_min_usd: float,
    skip_stables: bool,
    do_dust_sweep: bool,
    report: Report,
) -> None:
    # Delete state files
    deleted = []
    for p in [POS_FILE, KPI_FILE]:
        if p.exists():
            p.unlink(missing_ok=True)
            deleted.append(str(p))
    # Optional: clear logs
    if LOG_DIR.exists() and LOG_DIR.is_dir():
        # non-recursive safe clear of known patterns
        for child in LOG_DIR.glob("*.log"):
            try:
                child.unlink(missing_ok=True)
                deleted.append(str(child))
            except Exception:
                pass

    report.files_deleted.extend(deleted)

    # Dust sweep note (actual sweeping is typically done by tools/dust_sweeper.py in your workflows)
    if do_dust_sweep:
        report.dust.append(DustAction(
            asset="*", amount=0.0, notional=0.0, skipped=False,
            reason=f"Delegate dust sweep to workflow (DUST_MIN_USD={dust_min_usd}, skip_stables={skip_stables})"
        ))
        report.notes.append("Dust sweep flagged — run tools/dust_sweeper.py in your workflow.")


def write_september_baseline(report: Report) -> None:
    baseline = {
        "created_at": now_iso(),
        "label": "September Good Baseline",
        "guards": {
            "UNIVERSE_TOP_K": int(env("UNIVERSE_TOP_K", "25") or "25"),
            "MAX_POSITIONS": int(env("MAX_POSITIONS", "3") or "3"),
            "MAX_BUYS_PER_RUN": int(env("MAX_BUYS_PER_RUN", "1") or "1"),
            "ROTATE_WHEN_FULL": to_bool(env("ROTATE_WHEN_FULL", "false")),
            "ROTATE_WHEN_CASH_SHORT": to_bool(env("ROTATE_WHEN_CASH_SHORT", "true")),
            "RESERVE_CASH_PCT": float(env("RESERVE_CASH_PCT", "5") or "5"),
            "MIN_SELL_USD": float(env("MIN_SELL_USD", "10") or "10"),
            "DUST_MIN_USD": float(env("DUST_MIN_USD", "2") or "2"),
            "DUST_SKIP_STABLES": to_bool(env("DUST_SKIP_STABLES", "true"), True),
        },
        "notes": [
            "Use DRY_RUN=ON to simulate.",
            "Keep rotation conservative during restore.",
            "Widen candidate pool (UNIVERSE_TOP_K) only after balances are stable.",
        ]
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / "baseline_config.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    report.files_written.append(str(path))
    report.notes.append("Wrote September baseline guard-pack to .state/baseline_config.json")


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reset / Restore CryptoBot state and holdings.")
    p.add_argument("--mode", choices=["FORCE_LIQUIDATE", "CLEAN_REINIT", "SEPTEMBER_BASELINE"],
                   default=os.getenv("RESET_MODE", "FORCE_LIQUIDATE"),
                   help="Reset mode.")
    p.add_argument("--dry-run", default=env("DRY_RUN", "ON"),
                   help="ON or OFF")
    p.add_argument("--min-sell-usd", type=float, default=float(env("MIN_SELL_USD", "10") or "10"))
    p.add_argument("--dust-min-usd", type=float, default=float(env("DUST_MIN_USD", "2") or "2"))
    p.add_argument("--skip-stables", default=env("DUST_SKIP_STABLES", "true"),
                   help="true/false: skip stables when liquidating or sweeping")
    p.add_argument("--include-dust", action="store_true",
                   help="Force-sell even tiny notionals below MIN_SELL_USD")
    p.add_argument("--do-dust-sweep", action="store_true",
                   help="When CLEAN_REINIT, also request a dust sweep by the workflow.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dry_run = to_bool(args.dry_run, True)
    skip_stables = to_bool(args.skip_stables, True)

    report = Report(
        started_at=now_iso(),
        finished_at=None,
        mode=args.mode,
        dry_run=dry_run,
        ccxt_available=_CCXT_AVAILABLE,
        summary={},
        sells=[],
        dust=[],
        files_deleted=[],
        files_written=[],
        notes=[]
    )

    print(f"=== CryptoBot RESET ===")
    print(f"Mode: {args.mode}")
    print(f"Dry run: {dry_run}  |  CCXT available: {_CCXT_AVAILABLE}")
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if args.mode == "FORCE_LIQUIDATE":
            if not _CCXT_AVAILABLE and not dry_run:
                raise RuntimeError("ccxt is not available, but live ops requested (DRY_RUN=OFF).")
            kraken = KrakenClient(dry_run=dry_run)
            force_liquidate(
                kraken=kraken,
                min_sell_usd=float(args.min_sell_usd),
                include_dust=bool(args.include_dust),
                skip_stables=skip_stables,
                report=report,
            )

        elif args.mode == "CLEAN_REINIT":
            clean_reinit(
                dust_min_usd=float(args.dust_min_usd),
                skip_stables=skip_stables,
                do_dust_sweep=bool(args.do_dust_sweep),
                report=report,
            )

        elif args.mode == "SEPTEMBER_BASELINE":
            # wipe then write baseline snapshot
            clean_reinit(
                dust_min_usd=float(args.dust_min_usd),
                skip_stables=skip_stables,
                do_dust_sweep=False,
                report=report,
            )
            write_september_baseline(report)

        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        # Summaries
        total_sold = sum(s.notional for s in report.sells if s.executed)
        total_skipped = sum(d.notional for d in report.dust if d.skipped)
        report.summary.update({
            "total_sell_actions": len(report.sells),
            "total_sold_usd": round(total_sold, 2),
            "dust_entries": len(report.dust),
            "dust_skipped_usd": round(total_skipped, 2),
        })

        return_code = 0

    except Exception as e:
        report.summary.update({"error": str(e)})
        print(f"[ERROR] {e}", file=sys.stderr)
        return_code = 1

    finally:
        report.finished_at = now_iso()
        report.write(REPORT_PATH)
        print(f"Report written to: {REPORT_PATH}")

    return return_code


if __name__ == "__main__":
    sys.exit(main())
