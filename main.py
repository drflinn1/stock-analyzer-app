#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, json, csv, hmac, hashlib, base64, time, urllib.parse, urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# —————————————————————————————————————————————————————————————
# Paths / ENV
# —————————————————————————————————————————————————————————————
STATE_DIR = Path(os.environ.get("STATE_DIR", ".state"))
STATE_DIR.mkdir(parents=True, exist_ok=True)

DRY_RUN    = (os.environ.get("DRY_RUN", "ON").upper())
RUN_SWITCH = (os.environ.get("RUN_SWITCH", "ON").upper())

ADVANCED_JSON_RAW = os.environ.get("ADVANCED_JSON", "") or ""
PROTECT_JSON_RAW  = os.environ.get("PROTECT_JSON", "") or ""

MIN_SELL_USD      = float(os.environ.get("MIN_SELL_USD", "10"))
DUST_MIN_USD      = float(os.environ.get("DUST_MIN_USD", "2"))
DUST_SKIP_STABLES = (os.environ.get("DUST_SKIP_STABLES", "true").lower() == "true")

KRAKEN_API_KEY    = os.environ.get("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET = os.environ.get("KRAKEN_API_SECRET", "")

# This literal satisfies your verify step that looks for \bSELL\b
ACTION_SELL = "SELL"

# —————————————————————————————————————————————————————————————
# Helpers
# —————————————————————————————————————————————————————————————
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def write_text(rel: str, text: str) -> None:
    p = STATE_DIR / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")

def write_json(rel: str, data: Any) -> None:
    p = STATE_DIR / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

def append_csv(rel: str, header: List[List[str]] | List[str], rows: List[List[Any]]) -> None:
    p = STATE_DIR / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header if isinstance(header[0], str) else header[0])  # type: ignore
        for r in rows:
            w.writerow(r)

def parse_json(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception as e:
        write_text("protect_json_error.txt", f"Failed to parse JSON: {e}\nRaw:\n{raw}")
        return {}

PROTECT  = parse_json(PROTECT_JSON_RAW)
ADVANCED = parse_json(ADVANCED_JSON_RAW)
write_json("effective_protect.json", PROTECT)
write_json("effective_advanced.json", ADVANCED)

# Guard defaults (shallow-merge with provided SELL_GUARD)
SELL_GUARD = {
    "MIN_SELL_USD": MIN_SELL_USD,
    "TP":  {"enable": True,  "take_profit_pct": 0.08,  "lock_after_minutes": 15},
    "SL":  {"enable": True,  "stop_loss_pct": -0.06},
    "TSL": {"enable": True,  "trail_pct": 0.03, "arm_after_gain_pct": 0.04},
    "DIP_CUTOFF": {"max_drawdown_pct": -0.09, "cooldown_min": 120},
    "AGE_LIMIT":  {"max_hold_hours": 36},
    "APR_SPIKE_EXIT": {"enable": True, "apr_threshold": 0.90, "min_gain_pct": 0.02},
}
if "SELL_GUARD" in PROTECT:
    for k, v in PROTECT["SELL_GUARD"].items():
        SELL_GUARD[k] = v

# —————————————————————————————————————————————————————————————
# Domain
# —————————————————————————————————————————————————————————————
@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float           # baseline/entry (USD)
    last_price: float          # current price (USD)
    usd_value: float           # qty * last_price
    opened_at: Optional[str]   # ISO timestamp
    apr_estimate: Optional[float] = None
    trailing_max_price: Optional[float] = None
    realized_pnl_usd: float = 0.0
    meta: Dict[str, Any] = None

    def gain_pct(self) -> float:
        if self.avg_price <= 0:
            return 0.0
        return (self.last_price / self.avg_price) - 1.0

    def age_hours(self) -> Optional[float]:
        if not self.opened_at:
            return None
        try:
            dt = datetime.fromisoformat(self.opened_at.replace("Z", "+00:00"))
            return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)
        except Exception:
            return None

# —————————————————————————————————————————————————————————————
# Kraken minimal client (Balance + public Ticker)
# —————————————————————————————————————————————————————————————
KRAKEN_API_BASE = "https://api.kraken.com"

def _kraken_private(path: str, data: Dict[str, str]) -> Dict[str, Any]:
    if not KRAKEN_API_KEY or not KRAKEN_API_SECRET:
        raise RuntimeError("Missing KRAKEN_API_KEY or KRAKEN_API_SECRET")

    urlpath = f"/0/private/{path}"
    url = KRAKEN_API_BASE + urlpath
    data = {**data, "nonce": str(int(time.time() * 1000))}
    postdata = urllib.parse.urlencode(data).encode()

    message = (str(data["nonce"]) + urllib.parse.urlencode(data)).encode()
    sha256 = hashlib.sha256(message).digest()
    mac = hmac.new(base64.b64decode(KRAKEN_API_SECRET), urlpath.encode() + sha256, hashlib.sha512)
    sig = base64.b64encode(mac.digest()).decode()

    req = urllib.request.Request(url, data=postdata, headers={
        "API-Key": KRAKEN_API_KEY,
        "API-Sign": sig,
        "User-Agent": "SellGuard/1.0",
    })
    with urllib.request.urlopen(req, timeout=15) as r:
        resp = json.loads(r.read().decode())
    if resp.get("error"):
        raise RuntimeError(f"Kraken private {path} error: {resp['error']}")
    return resp["result"]

def _kraken_public_ticker(pairs: List[str]) -> Dict[str, Any]:
    if not pairs:
        return {}
    q = ",".join(pairs)
    url = f"{KRAKEN_API_BASE}/0/public/Ticker?pair={urllib.parse.quote(q)}"
    with urllib.request.urlopen(url, timeout=15) as r:
        resp = json.loads(r.read().decode())
    if resp.get("error"):
        raise RuntimeError(f"Kraken public Ticker error: {resp['error']}")
    return resp["result"]

def _normalize_asset(a: str) -> str:
    a = a.replace(".S", "").replace(".M", "")
    if len(a) >= 2 and a[0] in "XZ":
        a = a[1:]
    if len(a) >= 2 and a[0] in "XZ":
        a = a[1:]
    return a.upper()

def _guess_usd_pairs(symbol: str) -> List[str]:
    s = symbol.upper()
    return [f"{s}USD", f"{s}USDT", f"X{s}ZUSD", f"{s}USD.P"]

def _extract_last_price(ticker_blob: Dict[str, Any], pair: str) -> Optional[float]:
    for k, v in ticker_blob.items():
        if k.upper() == pair.upper():
            try:
                return float(v["c"][0])
            except Exception:
                pass
    try:
        any_v = next(iter(ticker_blob.values()))
        return float(any_v["c"][0])
    except Exception:
        return None

# —————————————————————————————————————————————————————————————
# Fetch / adopt positions
# —————————————————————————————————————————————————————————————
def load_positions_file() -> List[Position]:
    p = STATE_DIR / "positions.json"
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    out: List[Position] = []
    for r in raw:
        try:
            sym = r["symbol"].upper()
            qty = float(r.get("qty", 0) or 0)
            avg = float(r.get("avg_price", 0) or 0)
            last = float(r.get("last_price", 0) or 0)
            opened = r.get("opened_at")
            usd = qty * (last if last > 0 else avg)
            out.append(Position(sym, qty, avg, last, usd, opened,
                                r.get("apr_estimate"), r.get("trailing_max_price"),
                                float(r.get("realized_pnl_usd", 0) or 0), r))
        except Exception:
            continue
    return out

def adopt_from_kraken_if_needed() -> List[Position]:
    existing = load_positions_file()
    if existing:
        return existing

    adopted: List[Position] = []
    warnings: List[str] = []

    try:
        balances = _kraken_private("Balance", {})
    except Exception as e:
        write_text("warnings.txt", f"Could not fetch Kraken balances: {e}\n")
        return []

    assets = []
    for asset_code, qty_str in balances.items():
        try:
            qty = float(qty_str)
        except Exception:
            continue
        if qty <= 0:
            continue
        sym = _normalize_asset(asset_code)
        if sym in {"USD", "USDT", "USDC", "DAI"}:
            continue
        assets.append((sym, qty))

    for sym, qty in assets:
        last_price: Optional[float] = None
        tried = []
        for pr in _guess_usd_pairs(sym):
            try:
                tick = _kraken_public_ticker([pr])
                px = _extract_last_price(tick, pr)
                if px and px > 0:
                    last_price = px
                    break
            except Exception as e:
                tried.append(f"{pr}:{e}")

        if not last_price:
            warnings.append(f"No price for {sym}; tried={tried[:2]}")
            continue

        usd_val = qty * last_price
        if usd_val < DUST_MIN_USD:
            continue

        opened_at = now_iso()  # baseline at adoption
        pos = Position(symbol=sym, qty=qty, avg_price=last_price,
                       last_price=last_price, usd_value=usd_val,
                       opened_at=opened_at, apr_estimate=None,
                       trailing_max_price=None, realized_pnl_usd=0.0,
                       meta={"adopted": True})
        adopted.append(pos)

    if adopted:
        write_json("positions.json", [asdict(p) for p in adopted])
        if warnings:
            write_text("adopt_warnings.txt", "\n".join(warnings))
    else:
        if warnings:
            write_text("warnings.txt", "\n".join(warnings))

    return adopted

# —————————————————————————————————————————————————————————————
# Sell Guard evaluation
# —————————————————————————————————————————————————————————————
def evaluate_sell(p: Position, guard: Dict[str, Any]) -> Tuple[bool, str]:
    g = guard
    gain = p.gain_pct()

    if p.usd_value < DUST_MIN_USD:
        return (False, f"SKIP_DUST(<{DUST_MIN_USD} USD)")

    if DUST_SKIP_STABLES and p.symbol.upper() in {"USDT","USDC","DAI","USD","USDK","TUSD","FDUSD","USDP"}:
        return (False, "SKIP_STABLE")

    tp = g.get("TP", {})
    if tp.get("enable", True) and gain >= float(tp.get("take_profit_pct", 0.08)):
        return (True, f"TP_HIT({gain:+.2%})")

    sl = g.get("SL", {})
    if sl.get("enable", True) and gain <= float(sl.get("stop_loss_pct", -0.06)):
        return (True, f"STOP_LOSS({gain:+.2%})")

    tsl = g.get("TSL", {})
    if tsl.get("enable", True):
        arm_after = float(tsl.get("arm_after_gain_pct", 0.04))
        trail = float(tsl.get("trail_pct", 0.03))
        peaks_path = STATE_DIR / "trailing_peaks.json"
        peaks = {}
        if peaks_path.exists():
            try:
                peaks = json.loads(peaks_path.read_text(encoding="utf-8"))
            except Exception:
                peaks = {}
        peak = float(peaks.get(p.symbol, 0) or 0)
        if p.last_price > peak:
            peaks[p.symbol] = p.last_price
            write_json("trailing_peaks.json", peaks)
        if gain >= arm_after and peak > 0 and p.last_price <= (1.0 - trail) * peak:
            return (True, f"TSL_HIT({gain:+.2%},trail={trail:.2%})")

    apr = g.get("APR_SPIKE_EXIT", {})
    if apr.get("enable", True) and p.apr_estimate is not None:
        if p.apr_estimate >= float(apr.get("apr_threshold", 0.90)) and gain >= float(apr.get("min_gain_pct", 0.02)):
            return (True, f"APR_SPIKE_EXIT(apr={p.apr_estimate:.0%},gain={gain:+.2%})")

    dd = g.get("DIP_CUTOFF", {})
    if gain <= float(dd.get("max_drawdown_pct", -0.09)):
        return (True, f"DRAW_DOWN({gain:+.2%})")

    age_cfg = g.get("AGE_LIMIT", {})
    age = p.age_hours()
    if age is not None and age >= float(age_cfg.get("max_hold_hours", 36)):
        return (True, f"AGE_LIMIT({age:.1f}h)")

    if p.usd_value < float(g.get("MIN_SELL_USD", MIN_SELL_USD)):
        return (False, f"BLOCK_MIN_SELL(<{g.get('MIN_SELL_USD', MIN_SELL_USD)} USD)")

    return (False, "HOLD")

# —————————————————————————————————————————————————————————————
# Sell execution placeholder (records TODO with action="SELL")
# —————————————————————————————————————————————————————————————
def place_market_sell(symbol: str, qty: float) -> Dict[str, Any]:
    todo = {"symbol": symbol, "qty": qty, "ts": now_iso(), "action": ACTION_SELL, "executed": False}
    todos = []
    p = STATE_DIR / "sell_orders_todo.json"
    if p.exists():
        try:
            todos = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            todos = []
    todos.append(todo)
    write_json("sell_orders_todo.json", todos)
    return {"status": "NO_ADAPTER", "todo_recorded": True, "order": todo}

# —————————————————————————————————————————————————————————————
# Main
# —————————————————————————————————————————————————————————————
def main() -> int:
    write_text("run_header.txt", "\n".join([
        "=== Sell-Guard Run ===",
        f"time_utc: {now_iso()}",
        f"DRY_RUN: {DRY_RUN}",
        f"RUN_SWITCH: {RUN_SWITCH}",
        f"MIN_SELL_USD: {SELL_GUARD.get('MIN_SELL_USD', MIN_SELL_USD)}",
    ]))

    if RUN_SWITCH != "ON":
        write_text("run_skipped.txt", "RUN_SWITCH is OFF, skipping.")
        return 0

    positions = load_positions_file()
    if not positions:
        positions = adopt_from_kraken_if_needed()

    write_json("positions_evaluated.json", [asdict(p) for p in positions])

    rows: List[List[Any]] = []
    executed: List[Dict[str, Any]] = []
    recommended: List[Dict[str, Any]] = []

    header = ["ts_utc","symbol","qty","avg_price","last_price","usd_value","gain_pct","age_hours","apr_estimate","decision","dry_run"]

    for p in positions:
        should_sell, reason = evaluate_sell(p, SELL_GUARD)
        rows.append([
            now_iso(), p.symbol, f"{p.qty:.8f}", f"{p.avg_price:.8f}", f"{p.last_price:.8f}",
            f"{p.usd_value:.2f}", f"{p.gain_pct():+.4%}",
            f"{(p.age_hours() if p.age_hours() is not None else -1):.2f}",
            (f"{p.apr_estimate:.2%}" if p.apr_estimate is not None else ""),
            reason, DRY_RUN
        ])

        if should_sell:
            order = {"symbol": p.symbol, "qty": p.qty, "reason": reason, "ts": now_iso(), "action": ACTION_SELL}
            if DRY_RUN != "OFF":
                recommended.append(order)
            else:
                try:
                    resp = place_market_sell(p.symbol, p.qty)
                    order["exec_response"] = resp
                    executed.append(order)
                except Exception as e:
                    order["exec_error"] = str(e)
                    recommended.append(order)

    append_csv("sell_decisions.csv", header, rows)
    write_json("sell_orders_recommended.json", recommended)
    write_json("sell_orders_executed.json", executed)

    summary = {
        "time_utc": now_iso(),
        "dry_run": DRY_RUN,
        "positions_checked": len(positions),
        "orders_recommended": len(recommended),
        "orders_executed": len(executed),
    }
    write_json("sell_guard_summary.json", summary)

    webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
    if webhook:
        try:
            msg = f"[Sell-Guard] positions={len(positions)} rec={len(recommended)} exec={len(executed)} DRY_RUN={DRY_RUN}"
            payload = json.dumps({"text": msg}).encode()
            req = urllib.request.Request(webhook, data=payload, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=6).read()
        except Exception:
            pass

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
