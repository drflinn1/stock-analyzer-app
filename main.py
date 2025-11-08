#!/usr/bin/env python3
"""
Main unified runner for Crypto Live workflows.

Features:
- Hourly 1-coin rotation BUY (lightweight)
- Force Sell (--force-sell) with SLIP_PCT fallback
- Dust Sweeper (--dust-sweep) for tiny balances
- Always writes .state/run_summary.json and .state/run_summary.md
- EXIT_ON_ERROR switch to keep workflows green while logging errors

Environment variables:
  DRY_RUN: "ON" | "OFF"
  BUY_USD: e.g., "25"
  TP_PCT:  e.g., "5"   (logged for transparency)
  SL_PCT:  e.g., "1"   (logged for transparency)
  SLOW_WINDOW_MIN: e.g., "60" (logged)
  UNIVERSE_PICK: "AUTO" (auto-pick) or a symbol like "SOLUSD"
  KRAKEN_API_KEY, KRAKEN_API_SECRET
  EXIT_ON_ERROR: "ON" | "OFF"  (default ON; OFF means exit 0 even if errors occur)

Force Sell flags (via UI -> env):
  --force-sell
  FORCE_SELL_SYMBOL: "ALL" or "SOLUSD" etc.
  FORCE_SELL_SLIP_PCT: default "3.0"

Dust Sweeper flags (via UI -> env):
  --dust-sweep
  DUST_MIN_USD: default "0.50"
  DUST_SLIP_PCT: default "3.0"
"""

from __future__ import annotations
import argparse
import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# ---------- State ----------
STATE_DIR = Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_JSON = STATE_DIR / "run_summary.json"
SUMMARY_MD = STATE_DIR / "run_summary.md"

# ---------- Utils ----------
def env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else str(v)

def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def usd(x: float) -> str:
    return f"${x:,.2f}"

def safe_float(s: Any, default: float = 0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default

def write_summary(summary: Dict[str, Any]) -> None:
    # JSON
    try:
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    except Exception as e:
        print(f"[WARN] Failed writing {SUMMARY_JSON}: {e}", file=sys.stderr)
    # MD
    try:
        lines = []
        lines.append("# Run Summary\n")
        lines.append(f"**When (UTC):** {summary.get('when','')}")
        lines.append(f"**Mode:** {'DRY-RUN' if summary.get('dry_run') else 'LIVE'}")
        info = summary.get("info") or {}
        if info:
            lines.append("\n## Info")
            for k, v in info.items():
                lines.append(f"- **{k}**: {v}")
        acts = summary.get("actions") or []
        if acts:
            lines.append("\n## Actions")
            for a in acts:
                lines.append(f"- {a}")
        errs = summary.get("errors") or []
        if errs:
            lines.append("\n## Errors")
            for e in errs:
                lines.append(f"- {e}")
        arts = summary.get("artifacts") or []
        if arts:
            lines.append("\n## Artifacts")
            for a in arts:
                lines.append(f"- {a}")
        SUMMARY_MD.write_text("\n".join(lines) + "\n")
    except Exception as e:
        print(f"[WARN] Failed writing {SUMMARY_MD}: {e}", file=sys.stderr)

# ---------- Kraken minimal client ----------
API_BASE = "https://api.kraken.com"

def _nonce() -> str:
    return str(int(time.time() * 1000))

def _kraken_headers(path: str, data: Dict[str, str], secret_b64: str, api_key: str) -> Dict[str, str]:
    """
    Correct Kraken signature:
      API-Sign = base64( HMAC-SHA512( base64decode(secret), path + SHA256(nonce + POSTDATA) ) )
    where POSTDATA is the full urlencoded body INCLUDING the nonce key/value.
    """
    postdata = urlencode(data)                # full body with nonce
    sha = hashlib.sha256((data["nonce"] + postdata).encode()).digest()
    mac = hmac.new(base64.b64decode(secret_b64), path.encode() + sha, hashlib.sha512)
    sig_b64 = base64.b64encode(mac.digest()).decode()
    return {
        "API-Key": api_key,
        "API-Sign": sig_b64,
        "User-Agent": "stock-analyzer-app/1.0 (urllib)",
        "Content-Type": "application/x-www-form-urlencoded",
    }

def kraken_public(method: str, params: Dict[str, str] | None = None) -> Dict[str, Any]:
    url = f"{API_BASE}/0/public/{method}"
    if params:
        url += "?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "stock-analyzer-app"})
    with urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())

def kraken_private(method: str, data: Dict[str, str]) -> Dict[str, Any]:
    api_key = env("KRAKEN_API_KEY")
    secret = env("KRAKEN_API_SECRET")
    if not api_key or not secret:
        raise RuntimeError("[LIVE] Missing Kraken API credentials.")
    path = f"/0/private/{method}"
    payload = {"nonce": _nonce(), **data}
    headers = _kraken_headers(path, payload, secret, api_key)
    body = urlencode(payload).encode()
    req = Request(f"{API_BASE}{path}", data=body, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=25) as r:
            txt = r.read().decode()
            out = json.loads(txt)
    except Exception as e:
        raise RuntimeError(f"HTTP error calling {method}: {e}")
    if isinstance(out, dict) and out.get("error"):
        # Bubble Kraken errors clearly for logs
        raise RuntimeError(f"{method} error: {out['error']}")
    return out

# ---------- Trading helpers ----------
def get_ticker_price(pair: str) -> float:
    res = kraken_public("Ticker", {"pair": pair})
    if res.get("error"):
        raise RuntimeError(f"Ticker error for {pair}: {res['error']}")
    result = res.get("result", {})
    first_key = next(iter(result.keys()))
    last_trade = result[first_key]["c"][0]
    return float(last_trade)

def get_balances() -> Dict[str, float]:
    res = kraken_private("Balance", {})
    out: Dict[str, float] = {}
    for k, v in res.get("result", {}).items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out

def add_order_market(pair: str, side: str, vol: float) -> Dict[str, Any]:
    return kraken_private("AddOrder", {
        "pair": pair,
        "type": side,           # "buy" or "sell"
        "ordertype": "market",
        "volume": f"{vol:.10f}",
        # You can add "oflags":"post" etc. if needed
    })

def add_order_limit(pair: str, side: str, vol: float, price: float) -> Dict[str, Any]:
    return kraken_private("AddOrder", {
        "pair": pair,
        "type": side,
        "ordertype": "limit",
        "price": f"{price:.8f}",
        "volume": f"{vol:.10f}",
    })

def usd_to_volume(pair: str, spend_usd: float) -> float:
    px = get_ticker_price(pair)
    if px <= 0:
        raise RuntimeError(f"Bad price for {pair}")
    return max(spend_usd / px, 0.0)

def guess_asset_from_pair(pair: str) -> str:
    u = pair.upper()
    if "USD" in u:
        return u.replace("USD","").replace("/","")
    if "USDT" in u:
        return u.replace("USDT","").replace("/","")
    return u

# ---------- Run actions ----------
@dataclass
class RunContext:
    dry_run: bool
    actions: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

def action_rotation(ctx: RunContext) -> None:
    """Select one coin and BUY."""
    buy_usd = safe_float(env("BUY_USD","25"), 25.0)
    tp_pct = safe_float(env("TP_PCT","5"), 5.0)
    sl_pct = safe_float(env("SL_PCT","1"), 1.0)
    window = env("SLOW_WINDOW_MIN","60")

    forced_raw = env("UNIVERSE_PICK","").strip()
    forced_upper = forced_raw.upper()
    if forced_upper in {"", "AUTO", "AUTO-PICK", "AUTOPICK", "ANY", "AUTOSELECT", "AUTO_SELECT"}:
        forced = ""
    else:
        forced = forced_upper

    ctx.info.update({
        "BUY_USD": buy_usd,
        "TP_PCT": tp_pct,
        "SL_PCT": sl_pct,
        "SLOW_WINDOW_MIN": window,
        "UNIVERSE_PICK": forced if forced else "(auto)",
    })

    pair = None
    if forced:
        pair = forced
    else:
        mfile = STATE_DIR / "momentum_candidates.csv"
        if mfile.exists():
            try:
                import csv
                with mfile.open() as f:
                    r = list(csv.DictReader(f))
                if r:
                    pair = (r[0].get("symbol") or "").upper().strip() or None
            except Exception as e:
                ctx.errors.append(f"Read momentum_candidates.csv failed: {e}")
    if not pair:
        pair = "SOLUSD"

    try:
        vol = usd_to_volume(pair, buy_usd)
        if ctx.dry_run:
            ctx.actions.append(f"[DRY] BUY {pair} ~{vol:.8f} worth {usd(buy_usd)}")
        else:
            res = add_order_market(pair, "buy", vol)
            txid = ",".join(res.get("result", {}).get("txid", []))
            ctx.actions.append(f"[LIVE] BUY {pair} {vol:.8f} ({usd(buy_usd)}) txid={txid}")
    except Exception as e:
        ctx.errors.append(str(e))

def action_force_sell(ctx: RunContext) -> None:
    symbol = env("FORCE_SELL_SYMBOL","ALL").strip().upper()
    slip_pct = safe_float(env("FORCE_SELL_SLIP_PCT","3.0"), 3.0)
    ctx.info.update({"FORCE_SELL_SYMBOL": symbol, "FORCE_SELL_SLIP_PCT": slip_pct})

    targets: List[str] = []
    if symbol == "ALL":
        try:
            bals = get_balances() if not ctx.dry_run else {"EXAMPLE": 1.0}
            for a, qty in bals.items():
                if qty <= 0:
                    continue
                if a.upper() in ("ZUSD","USD","USDT","ZUSDT"):
                    continue
                guess = f"{a.replace('X','').replace('Z','')}USD"
                targets.append(guess)
        except Exception as e:
            ctx.errors.append(f"Balance read failed: {e}")
            return
    else:
        targets = [symbol]

    for pair in targets:
        try:
            base_asset = guess_asset_from_pair(pair)
            if ctx.dry_run:
                ctx.actions.append(f"[DRY] SELL {pair} (force) all-in (vol≈?)")
                continue

            bals = get_balances()
            candidates = [base_asset, f"X{base_asset}", f"Z{base_asset}", base_asset.replace("X","").replace("Z","")]
            qty = 0.0
            for k in candidates:
                if k in bals and bals[k] > 0:
                    qty = bals[k]
                    break
            if qty <= 0:
                ctx.actions.append(f"[LIVE] No balance found for {base_asset} -> skip {pair}")
                continue

            try:
                res = add_order_market(pair, "sell", qty)
                txid = ",".join(res.get("result", {}).get("txid", []))
                ctx.actions.append(f"[LIVE] SELL {pair} {qty:.8f} (market) txid={txid}")
            except Exception as e:
                px = get_ticker_price(pair)
                limit_px = px * (1.0 - slip_pct/100.0)
                res2 = add_order_limit(pair, "sell", qty, limit_px)
                if res2.get("error"):
                    ctx.errors.append(f"SELL {pair} failed (limit retry): {res2['error']}; original market error: {e}")
                else:
                    txid2 = ",".join(res2.get("result", {}).get("txid", []))
                    ctx.actions.append(f"[LIVE] SELL {pair} {qty:.8f} (limit {limit_px:.8f}) txid={txid2} (market blocked)")
        except Exception as e:
            ctx.errors.append(f"{pair}: {e}")

def action_dust_sweep(ctx: RunContext) -> None:
    min_usd = safe_float(env("DUST_MIN_USD","0.50"), 0.50)
    slip_pct = safe_float(env("DUST_SLIP_PCT","3.0"), 3.0)
    ctx.info.update({"DUST_MIN_USD": min_usd, "DUST_SLIP_PCT": slip_pct})

    try:
        if ctx.dry_run:
            ctx.actions.append(f"[DRY] Would sweep positions under {usd(min_usd)}")
            return

        bals = get_balances()
        for a, qty in bals.items():
            if qty <= 0:
                continue
            if a.upper() in ("ZUSD","USD","USDT","ZUSDT"):
                continue
            pair = f"{a.replace('X','').replace('Z','')}USD"
            try:
                px = get_ticker_price(pair)
            except Exception as e:
                ctx.errors.append(f"Skip {a}: cannot price {pair}: {e}")
                continue
            value = qty * px
            if value < min_usd:
                try:
                    res = add_order_market(pair, "sell", qty)
                    txid = ",".join(res.get("result", {}).get("txid", []))
                    ctx.actions.append(f"[LIVE] DUST SELL {pair} {qty:.8f} (~{usd(value)}) market txid={txid}")
                except Exception as e:
                    limit_px = px * (1.0 - slip_pct/100.0)
                    res2 = add_order_limit(pair, "sell", qty, limit_px)
                    if res2.get("error"):
                        ctx.errors.append(f"DUST {pair} failed: {res2['error']}; original market error: {e}")
                    else:
                        txid2 = ",".join(res2.get("result", {}).get("txid", []))
                        ctx.actions.append(f"[LIVE] DUST SELL {pair} {qty:.8f} (~{usd(value)}) limit {limit_px:.8f} txid={txid2} (market blocked)")
        if not ctx.actions:
            ctx.actions.append("[LIVE] Dust sweep found nothing under threshold.")
    except Exception as e:
        ctx.errors.append(str(e))

# ---------- Entry ----------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-sell", action="store_true", help="Force sell symbol(s)")
    parser.add_argument("--dust-sweep", action="store_true", help="Sell tiny positions under MIN_USD")
    args = parser.parse_args()

    dry = env("DRY_RUN","ON").upper().strip() != "OFF"
    summary = RunContext(dry_run=dry)
    summary.info["when"] = now_iso()
    summary.info["runner"] = "main.py v1.0c (fixed Kraken signing)"

    try:
        if args.force_sell:
            action_force_sell(summary)
        elif args.dust_sweep:
            action_dust_sweep(summary)
        else:
            action_rotation(summary)
    except Exception as e:
        summary.errors.append(f"Top-level error: {e}")

    out = {
        "when": now_iso(),
        "dry_run": summary.dry_run,
        "info": summary.info,
        "actions": summary.actions,
        "errors": summary.errors,
        "artifacts": [str(SUMMARY_JSON), str(SUMMARY_MD)],
    }
    write_summary(out)

    # Keep job green if EXIT_ON_ERROR=OFF (default ON)
    exit_on_error = env("EXIT_ON_ERROR","ON").upper().strip() == "ON"
    has_errors = bool(summary.errors)
    if summary.dry_run:
        return 0
    return 1 if (exit_on_error and has_errors) else 0

if __name__ == "__main__":
    sys.exit(main())
