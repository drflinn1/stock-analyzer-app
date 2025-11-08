#!/usr/bin/env python3
"""
Main unified runner for Crypto Live workflows (Kraken signing hardened).

- Hourly 1-coin rotation BUY
- Force Sell (--force-sell) with SLIP_PCT fallback
- Dust Sweeper (--dust-sweep)
- Always writes .state/run_summary.json/.md
- EXIT_ON_ERROR: default ON

Env:
  DRY_RUN ("ON"/"OFF"), BUY_USD, TP_PCT, SL_PCT, SLOW_WINDOW_MIN, UNIVERSE_PICK
  KRAKEN_API_KEY, KRAKEN_API_SECRET
"""

from __future__ import annotations
import argparse, base64, datetime as dt, hashlib, hmac, json, os, sys, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# ---------- State ----------
STATE_DIR = Path(".state"); STATE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_JSON = STATE_DIR / "run_summary.json"
SUMMARY_MD   = STATE_DIR / "run_summary.md"

# ---------- Utils ----------
def env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else str(v)

def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat()+"Z"

def usd(x: float) -> str:
    return f"${x:,.2f}"

def safe_float(s: Any, default: float = 0.0) -> float:
    try: return float(s)
    except Exception: return default

def write_summary(summary: Dict[str, Any]) -> None:
    try: SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    except Exception as e: print(f"[WARN] write {SUMMARY_JSON}: {e}", file=sys.stderr)
    try:
        lines = ["# Run Summary\n",
                 f"**When (UTC):** {summary.get('when','')}",
                 f"**Mode:** {'DRY-RUN' if summary.get('dry_run') else 'LIVE'}"]
        info = summary.get("info") or {}
        if info:
            lines.append("\n## Info"); lines += [f"- **{k}**: {v}" for k,v in info.items()]
        acts = summary.get("actions") or []
        if acts:
            lines.append("\n## Actions"); lines += [f"- {a}" for a in acts]
        errs = summary.get("errors") or []
        if errs:
            lines.append("\n## Errors"); lines += [f"- {e}" for e in errs]
        arts = summary.get("artifacts") or []
        if arts:
            lines.append("\n## Artifacts"); lines += [f"- {a}" for a in arts]
        SUMMARY_MD.write_text("\n".join(lines)+"\n")
    except Exception as e:
        print(f"[WARN] write {SUMMARY_MD}: {e}", file=sys.stderr)

# ---------- Kraken client (REST) ----------
API_BASE = "https://api.kraken.com"

def _nonce() -> str:
    return str(int(time.time()*1000))

def _clean_key_secret() -> tuple[str, bytes, list[str]]:
    msgs: List[str] = []
    api_key = env("KRAKEN_API_KEY","").strip()
    secret_b64 = env("KRAKEN_API_SECRET","").strip()
    if not api_key or not secret_b64:
        raise RuntimeError("[LIVE] Missing Kraken API credentials.")
    try:
        secret_raw = base64.b64decode(secret_b64, validate=True)
        msgs.append(f"kraken_secret_decoded_len={len(secret_raw)}")
    except Exception as e:
        raise RuntimeError(f"[LIVE] Kraken secret is not valid base64 ({e}). Recreate/copy again.")
    msgs.append(f"kraken_api_key_len={len(api_key)}")
    return api_key, secret_raw, msgs

def _kraken_headers(path: str, data: Dict[str,str], api_key: str, secret_raw: bytes) -> Dict[str,str]:
    # Per Kraken: HMAC-SHA512( secret, path + SHA256(nonce + POSTDATA) )
    postdata = urlencode(data)                       # full body (includes nonce)
    sha = hashlib.sha256((data["nonce"] + postdata).encode()).digest()
    sig = hmac.new(secret_raw, path.encode()+sha, hashlib.sha512).digest()
    return {
        "API-Key": api_key,
        "API-Sign": base64.b64encode(sig).decode(),
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "stock-analyzer-app/urllib",
    }

def kraken_public(method: str, params: Dict[str,str]|None=None) -> Dict[str,Any]:
    url = f"{API_BASE}/0/public/{method}"
    if params: url += "?"+urlencode(params)
    req = Request(url, headers={"User-Agent":"stock-analyzer-app"})
    with urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())

def kraken_private(method: str, data: Dict[str,str]) -> Dict[str,Any]:
    api_key, secret_raw, msg = _clean_key_secret()
    # lightweight diagnostics written to summary later
    path = f"/0/private/{method}"
    payload = {"nonce": _nonce(), **data}
    headers = _kraken_headers(path, payload, api_key, secret_raw)
    body = urlencode(payload).encode()
    req = Request(f"{API_BASE}{path}", data=body, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=25) as r:
            out = json.loads(r.read().decode())
    except Exception as e:
        raise RuntimeError(f"{method} HTTP error: {e}")
    if isinstance(out, dict) and out.get("error"):
        # Bubble Kraken errors clearly
        raise RuntimeError(f"{method} error: {out['error']}  ({'; '.join(msg)})")
    return out

# ---------- Trading helpers ----------
def get_ticker_price(pair: str) -> float:
    res = kraken_public("Ticker", {"pair": pair})
    if res.get("error"): raise RuntimeError(f"Ticker error {pair}: {res['error']}")
    result = res.get("result", {}); first = next(iter(result.keys()))
    return float(result[first]["c"][0])

def get_balances() -> Dict[str,float]:
    res = kraken_private("Balance", {})
    out: Dict[str,float] = {}
    for k,v in res.get("result",{}).items():
        try: out[k]=float(v)
        except Exception: pass
    return out

def add_order_market(pair: str, side: str, vol: float) -> Dict[str,Any]:
    return kraken_private("AddOrder", {"pair":pair,"type":side,"ordertype":"market","volume":f"{vol:.10f}"})

def add_order_limit(pair: str, side: str, vol: float, price: float) -> Dict[str,Any]:
    return kraken_private("AddOrder", {"pair":pair,"type":side,"ordertype":"limit","price":f"{price:.8f}","volume":f"{vol:.10f}"})

def usd_to_volume(pair: str, spend_usd: float) -> float:
    px = get_ticker_price(pair)
    if px <= 0: raise RuntimeError(f"Bad price for {pair}")
    return max(spend_usd/px, 0.0)

def guess_asset_from_pair(pair: str) -> str:
    u = pair.upper()
    if "USD" in u:  return u.replace("USD","").replace("/","")
    if "USDT" in u: return u.replace("USDT","").replace("/","")
    return u

# ---------- Run actions ----------
@dataclass
class RunContext:
    dry_run: bool
    actions: List[str] = field(default_factory=list)
    errors:  List[str] = field(default_factory=list)
    info:    Dict[str,Any] = field(default_factory=dict)

def action_rotation(ctx: RunContext) -> None:
    buy_usd = safe_float(env("BUY_USD","25"), 25.0)
    tp_pct  = safe_float(env("TP_PCT","5"), 5.0)
    sl_pct  = safe_float(env("SL_PCT","1"), 1.0)
    window  = env("SLOW_WINDOW_MIN","60")
    forced_raw = env("UNIVERSE_PICK","").strip().upper()
    forced = "" if forced_raw in {"","AUTO","AUTO-PICK","AUTOPICK","ANY","AUTOSELECT","AUTO_SELECT"} else forced_raw
    ctx.info.update({"BUY_USD":buy_usd,"TP_PCT":tp_pct,"SL_PCT":sl_pct,"SLOW_WINDOW_MIN":window,"UNIVERSE_PICK":forced or "(auto)"})

    pair = forced or "SOLUSD"
    if not forced:
        m = STATE_DIR / "momentum_candidates.csv"
        if m.exists():
            try:
                import csv; rows = list(csv.DictReader(m.open()))
                if rows: pair = (rows[0].get("symbol") or "").upper().strip() or pair
            except Exception as e:
                ctx.errors.append(f"Read momentum_candidates.csv failed: {e}")

    try:
        vol = usd_to_volume(pair, buy_usd)
        if ctx.dry_run:
            ctx.actions.append(f"[DRY] BUY {pair} ~{vol:.8f} worth {usd(buy_usd)}")
        else:
            res = add_order_market(pair, "buy", vol)
            txid = ",".join(res.get("result",{}).get("txid",[]))
            ctx.actions.append(f"[LIVE] BUY {pair} {vol:.8f} ({usd(buy_usd)}) txid={txid}")
    except Exception as e:
        ctx.errors.append(str(e))

def action_force_sell(ctx: RunContext) -> None:
    symbol  = env("FORCE_SELL_SYMBOL","ALL").strip().upper()
    slip_pct= safe_float(env("FORCE_SELL_SLIP_PCT","3.0"),3.0)
    ctx.info.update({"FORCE_SELL_SYMBOL":symbol,"FORCE_SELL_SLIP_PCT":slip_pct})

    targets: List[str] = []
    if symbol=="ALL":
        try:
            bals = get_balances() if not ctx.dry_run else {"EXAMPLE":1.0}
            for a,qty in bals.items():
                if qty<=0 or a.upper() in ("ZUSD","USD","USDT","ZUSDT"): continue
                targets.append(f"{a.replace('X','').replace('Z','')}USD")
        except Exception as e:
            ctx.errors.append(f"Balance read failed: {e}"); return
    else:
        targets=[symbol]

    for pair in targets:
        try:
            base = guess_asset_from_pair(pair)
            if ctx.dry_run:
                ctx.actions.append(f"[DRY] SELL {pair} (force) all-in (vol≈?)"); continue
            bals = get_balances()
            cand = [base, f"X{base}", f"Z{base}", base.replace("X","").replace("Z","")]
            qty = next((bals[k] for k in cand if k in bals and bals[k]>0), 0.0)
            if qty<=0:
                ctx.actions.append(f"[LIVE] No balance for {base} -> skip {pair}"); continue
            try:
                res = add_order_market(pair,"sell",qty)
                txid = ",".join(res.get("result",{}).get("txid",[]))
                ctx.actions.append(f"[LIVE] SELL {pair} {qty:.8f} (market) txid={txid}")
            except Exception as e:
                px = get_ticker_price(pair); limit_px = px*(1.0-slip_pct/100.0)
                res2 = add_order_limit(pair,"sell",qty,limit_px)
                if res2.get("error"):
                    ctx.errors.append(f"SELL {pair} failed (limit retry): {res2['error']}; original: {e}")
                else:
                    txid2 = ",".join(res2.get("result",{}).get("txid",[]))
                    ctx.actions.append(f"[LIVE] SELL {pair} {qty:.8f} (limit {limit_px:.8f}) txid={txid2} (market blocked)")
        except Exception as e:
            ctx.errors.append(f"{pair}: {e}")

def action_dust_sweep(ctx: RunContext) -> None:
    min_usd = safe_float(env("DUST_MIN_USD","0.50"),0.50)
    slip_pct= safe_float(env("DUST_SLIP_PCT","3.0"),3.0)
    ctx.info.update({"DUST_MIN_USD":min_usd,"DUST_SLIP_PCT":slip_pct})
    try:
        if ctx.dry_run:
            ctx.actions.append(f"[DRY] Would sweep positions under {usd(min_usd)}"); return
        bals = get_balances()
        for a,qty in bals.items():
            if qty<=0 or a.upper() in ("ZUSD","USD","USDT","ZUSDT"): continue
            pair = f"{a.replace('X','').replace('Z','')}USD"
            try: px = get_ticker_price(pair)
            except Exception as e: ctx.errors.append(f"Skip {a}: cannot price {pair}: {e}"); continue
            value = qty*px
            if value < min_usd:
                try:
                    res = add_order_market(pair,"sell",qty)
                    txid = ",".join(res.get("result",{}).get("txid",[]))
                    ctx.actions.append(f"[LIVE] DUST SELL {pair} {qty:.8f} (~{usd(value)}) market txid={txid}")
                except Exception as e:
                    limit_px = px*(1.0-slip_pct/100.0)
                    res2 = add_order_limit(pair,"sell",qty,limit_px)
                    if res2.get("error"):
                        ctx.errors.append(f"DUST {pair} failed: {res2['error']}; original: {e}")
                    else:
                        txid2 = ",".join(res2.get("result",{}).get("txid",[]))
                        ctx.actions.append(f"[LIVE] DUST SELL {pair} {qty:.8f} (~{usd(value)}) limit {limit_px:.8f} txid={txid2} (market blocked)")
        if not ctx.actions: ctx.actions.append("[LIVE] Dust sweep found nothing under threshold.")
    except Exception as e:
        ctx.errors.append(str(e))

# ---------- Entry ----------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-sell", action="store_true")
    parser.add_argument("--dust-sweep", action="store_true")
    args = parser.parse_args()

    dry = env("DRY_RUN","ON").upper().strip() != "OFF"
    summary = RunContext(dry_run=dry)
    summary.info["when"]   = now_iso()
    summary.info["runner"] = "main.py v1.0d (signing + trimming + decode check)"

    try:
        if args.force_sell:   action_force_sell(summary)
        elif args.dust_sweep: action_dust_sweep(summary)
        else:                 action_rotation(summary)
    except Exception as e:
        summary.errors.append(f"Top-level error: {e}")

    out = {"when": now_iso(), "dry_run": summary.dry_run, "info": summary.info,
           "actions": summary.actions, "errors": summary.errors,
           "artifacts": [str(SUMMARY_JSON), str(SUMMARY_MD)]}
    write_summary(out)

    exit_on_error = env("EXIT_ON_ERROR","ON").upper().strip() == "ON"
    return 0 if (summary.dry_run or not (exit_on_error and summary.errors)) else 1

if __name__ == "__main__":
    sys.exit(main())
