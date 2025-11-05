#!/usr/bin/env python3
"""
Crypto — Hourly 1-Coin Rotation (LIVE-ready, guarded)
- One open position enforced via .state/position.json
- Exits: SL (default 1%), TP (default 5%), Timer (60m & gain < +3%)
- ALWAYS writes .state/run_summary.json/.md
- Minimal Kraken REST client (stdlib only)

Adds:
- Sanity clamps on envs (no negative SL, etc.)
- Buy-guard: prevents a new BUY if a recent buy occurred within BUY_GUARD_MINUTES (default 5)
"""

from __future__ import annotations
import os, json, time, datetime as dt
from pathlib import Path
from typing import Dict, Any, Optional

# ------------------------------- Paths ---------------------------------
STATE = Path(".state"); STATE.mkdir(parents=True, exist_ok=True)
POS_FILE = STATE / "position.json"
SUMMARY_JSON = STATE / "run_summary.json"
SUMMARY_MD   = STATE / "run_summary.md"
LAST_BUY_TS  = STATE / "last_buy_at.txt"   # stores epoch seconds of last live/dry buy

# ------------------------------ Env utils ------------------------------
def env_str(k: str, default: str = "") -> str:
    v = os.getenv(k, default)
    return "" if v is None else str(v)

def env_int(k: str, default: int = 0) -> int:
    try: return int(env_str(k, str(default)).strip())
    except: return default

def env_float(k: str, default: float = 0.0) -> float:
    try: return float(env_str(k, str(default)).strip())
    except: return default

def clamp(v: float, lo: float, hi: float, default: float) -> float:
    try:
        if v < lo or v > hi or not (v == v):  # NaN check v==v
            return default
        return v
    except Exception:
        return default

def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# --------------------------- Summary writers ---------------------------
def _save_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))

def _load_json(p: Path) -> Optional[Any]:
    if p.exists():
        return json.loads(p.read_text())
    return None

def write_summary(status: str, details: Dict[str, Any]) -> None:
    payload = {
        "when": now_iso(),
        "mode": "LIVE" if env_str("DRY_RUN","OFF").upper() == "OFF" else "DRY_RUN",
        "status": status,
        **details
    }
    _save_json(SUMMARY_JSON, payload)
    lines = [
        f"**When:** {payload['when']}",
        f"**Mode:** {payload['mode']}",
        f"**Status:** {status}",
        "",
        "```json",
        json.dumps(details, indent=2),
        "```",
    ]
    SUMMARY_MD.write_text("\n".join(lines))

# ---------------------------- State helpers ----------------------------
def have_open_position() -> bool:
    return POS_FILE.exists()

def load_position() -> Optional[Dict[str, Any]]:
    return _load_json(POS_FILE)

def save_position(symbol: str, qty: float, entry_price: float) -> None:
    _save_json(POS_FILE, {
        "symbol": symbol,
        "qty": float(qty),
        "entry_price": float(entry_price),
        "opened_at": time.time()
    })

def clear_position() -> None:
    if POS_FILE.exists():
        POS_FILE.unlink()

def pct_change(from_price: float, to_price: float) -> float:
    if from_price <= 0: return 0.0
    return (to_price - from_price) / from_price * 100.0

def record_last_buy_now() -> None:
    LAST_BUY_TS.write_text(str(int(time.time())))

def minutes_since_last_buy() -> Optional[float]:
    if not LAST_BUY_TS.exists():
        return None
    try:
        ts = int(LAST_BUY_TS.read_text().strip() or "0")
        return (time.time() - ts) / 60.0
    except Exception:
        return None

# ------------------------ Candidate selection --------------------------
def read_csv_first_symbol(csv_path: Path) -> Optional[str]:
    try:
        if not csv_path.exists(): return None
        rows = csv_path.read_text().strip().splitlines()
        if not rows: return None
        hdr = rows[0].lower().replace(" ", "")
        start = 1 if ("symbol" in hdr or "," in hdr) else 0
        for line in rows[start:]:
            if not line.strip(): continue
            sym = line.split(",")[0].strip()
            if sym: return sym
    except Exception:
        return None
    return None

def normalize_pair(sym: str) -> str:
    s = sym.upper().replace("-", "").replace("/", "")
    if s.endswith("USD"): return s
    return s + "USD"

def select_top_candidate() -> str:
    for name in ("momentum_candidates.csv", "spike_candidates.csv"):
        sym = read_csv_first_symbol(STATE / name)
        if sym: return normalize_pair(sym)
    up = env_str("UNIVERSE_PICK", "").strip()
    if up: return normalize_pair(up)
    raise RuntimeError("No candidate found (candidates CSV and UNIVERSE_PICK empty).")

# -------------------------- Kraken REST client -------------------------
import base64, hashlib, hmac, urllib.parse, urllib.request

API_BASE = "https://api.kraken.com"
KRAKEN_API_KEY    = env_str("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET = env_str("KRAKEN_API_SECRET", "")

def _http_get(url: str, timeout: int = 20) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())

def _private_request(path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    if not KRAKEN_API_KEY or not KRAKEN_API_SECRET:
        raise RuntimeError("Missing KRAKEN_API_KEY or KRAKEN_API_SECRET")
    url = API_BASE + path
    data = dict(data)
    data["nonce"] = str(int(time.time() * 1000))
    postdata = urllib.parse.urlencode(data)
    sha = hashlib.sha256((data["nonce"] + postdata).encode()).digest()
    msg = path.encode() + sha
    sig = hmac.new(base64.b64decode(KRAKEN_API_SECRET), msg, hashlib.sha512).digest()
    headers = {
        "API-Key": KRAKEN_API_KEY,
        "API-Sign": base64.b64encode(sig).decode(),
        "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        "User-Agent": "rotation-bot/1.0",
    }
    req = urllib.request.Request(url, data=postdata.encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode())

def kraken_last_price(pair: str) -> float:
    url = f"{API_BASE}/0/public/Ticker?pair={pair}"
    data = _http_get(url)
    if data.get("error"):
        raise RuntimeError(f"Ticker error: {data['error']}")
    result = data.get("result", {})
    if not result:
        raise RuntimeError("Ticker empty result")
    first = next(iter(result.values()))
    price = float(first["c"][0])
    return price

def _volume_round(qty: float) -> float:
    q = float(f"{qty:.8f}")
    return 0.0 if q < 1e-8 else q

def kraken_add_order_market(pair: str, side: str, volume_base: float) -> Dict[str, Any]:
    """
    side: 'buy' or 'sell'
    ordertype: 'market'
    volume_base: BASE units (e.g., AERO units)
    """
    payload = {
        "pair": pair,
        "type": side,
        "ordertype": "market",
        "volume": f"{_volume_round(volume_base):.8f}",
    }
    return _private_request("/0/private/AddOrder", payload)

# -------------------------- Live buy/sell API --------------------------
def get_last_price(symbol: str) -> float:
    return kraken_last_price(symbol)

def live_buy(symbol: str, usd_amount: float) -> Dict[str, Any]:
    price = kraken_last_price(symbol)
    if price <= 0:
        raise RuntimeError(f"Bad price for {symbol}: {price}")
    qty = usd_amount / price
    qty = _volume_round(qty)
    if qty <= 0:
        raise RuntimeError(f"Calculated qty too small for {symbol} at {price} using ${usd_amount}")
    res = kraken_add_order_market(symbol, "buy", qty)
    err = res.get("error", [])
    if err:
        raise RuntimeError(f"KRAKEN BUY error: {err}")
    txs  = res.get("result", {}).get("txid", [])
    return {"symbol": symbol, "qty": qty, "avg_price": price, "txid": txs[0] if txs else None, "raw": res}

def live_sell(symbol: str, qty: float) -> Dict[str, Any]:
    qty = _volume_round(qty)
    if qty <= 0:
        raise RuntimeError(f"SELL qty too small for {symbol}")
    res = kraken_add_order_market(symbol, "sell", qty)
    err = res.get("error", [])
    if err:
        raise RuntimeError(f"KRAKEN SELL error: {err}")
    price = kraken_last_price(symbol)
    txs  = res.get("result", {}).get("txid", [])
    return {"symbol": symbol, "qty": qty, "avg_price": price, "txid": txs[0] if txs else None, "raw": res}

# ------------------------------ Main logic ------------------------------
def main() -> None:
    # Raw envs
    raw_dry_run        = env_str("DRY_RUN","OFF").upper()
    run_switch         = env_str("RUN_SWITCH","ON").upper() == "ON"
    usd_per_buy        = env_float("BUY_USD", 25.0)
    max_positions      = env_int("MAX_POSITIONS", 1)
    tp_pct_raw         = env_float("TP_PCT", 5.0)
    sl_pct_raw         = env_float("SL_PCT", 1.0)
    rotate_minutes_raw = env_int("ROTATE_MINUTES", 60)
    rotate_gain_raw    = env_float("ROTATE_MIN_GAIN_PCT", 3.0)
    buy_guard_min      = env_int("BUY_GUARD_MINUTES", 5)  # optional guard

    # Sanity clamps
    tp_pct         = clamp(tp_pct_raw, 0.1, 100.0, 5.0)
    sl_pct         = clamp(sl_pct_raw, 0.1, 50.0, 1.0)
    rotate_minutes = int(clamp(float(rotate_minutes_raw), 5.0, 480.0, 60.0))
    rotate_gain    = clamp(rotate_gain_raw, 0.0, 20.0, 3.0)
    dry_run        = (raw_dry_run != "OFF")

    write_summary("start", {
        "dry_run": dry_run,
        "run_switch": run_switch,
        "max_positions": max_positions,
        "buy_usd": usd_per_buy,
        "tp_pct": {"raw": tp_pct_raw, "sanitized": tp_pct},
        "sl_pct": {"raw": sl_pct_raw, "sanitized": sl_pct},
        "rotate_minutes": {"raw": rotate_minutes_raw, "sanitized": rotate_minutes},
        "rotate_gain_pct": {"raw": rotate_gain_raw, "sanitized": rotate_gain},
        "buy_guard_minutes": buy_guard_min
    })

    if not run_switch:
        write_summary("SKIP", {"reason":"RUN_SWITCH=OFF"})
        return

    # If holding, evaluate exits
    if have_open_position():
        pos = load_position()
        if not pos:
            clear_position()
            write_summary("WARN", {"reason":"position.json unreadable; cleared"})
            return

        sym = pos["symbol"]; qty = float(pos["qty"])
        entry = float(pos["entry_price"]); opened = float(pos["opened_at"])
        price = get_last_price(sym)
        change = ((price - entry) / entry * 100.0) if entry > 0 else 0.0
        minutes = (time.time() - opened) / 60.0

        decision, reason = "hold", ""
        if change <= -abs(sl_pct):
            decision, reason = "sell", f"SL hit ({change:.2f}%)"
        elif change >= abs(tp_pct):
            decision, reason = "sell", f"TP hit (+{change:.2f}%)"
        elif minutes >= rotate_minutes and change < abs(rotate_gain):
            decision, reason = "sell", f"Timer {rotate_minutes}m & gain {change:.2f}% < {rotate_gain}%"

        if decision == "sell":
            if dry_run:
                write_summary("DRY SELL", {"symbol": sym, "qty": qty, "price": price, "reason": reason})
                clear_position()
            else:
                try:
                    res = live_sell(sym, qty)
                    write_summary("LIVE SELL OK", {
                        "symbol": res["symbol"], "qty": res["qty"],
                        "avg_price": res["avg_price"], "txid": res.get("txid"),
                        "reason": reason
                    })
                    clear_position()
                except Exception as e:
                    write_summary("LIVE SELL ERROR", {"symbol": sym, "qty": qty, "exception": repr(e)})
        else:
            write_summary("HOLD", {"symbol": sym, "qty": qty, "price": price, "change_pct": change, "minutes_open": minutes})
        return

    # Not holding: enforce caps / guards, then BUY
    if max_positions <= 0:
        write_summary("SKIP", {"reason":"MAX_POSITIONS<=0"})
        return
    if max_positions == 1 and POS_FILE.exists():
        write_summary("SKIP", {"reason":"position.json present; MAX_POSITIONS=1"})
        return

    # Buy-guard: avoid rapid-fire rebuy after manual reruns / timing races
    since = minutes_since_last_buy()
    if since is not None and since < max(1, buy_guard_min):
        write_summary("SKIP", {"reason": f"Buy guard: last buy {since:.1f}m ago < {buy_guard_min}m"})
        return

    # Choose candidate
    try:
        pick = select_top_candidate()
    except Exception as e:
        write_summary("SKIP", {"reason": f"No candidate: {e}"})
        return

    # Execute BUY
    try:
        if dry_run:
            price = get_last_price(pick)
            qty = (usd_per_buy / price) if price > 0 else 0.0
            qty = float(f"{qty:.8f}")
            save_position(pick, qty, price)
            record_last_buy_now()
            write_summary("DRY BUY", {"symbol": pick, "qty": qty, "entry_price": price})
        else:
            res = live_buy(pick, usd_per_buy)
            save_position(res["symbol"], float(res["qty"]), float(res["avg_price"]))
            record_last_buy_now()
            write_summary("LIVE BUY OK", {
                "symbol": res["symbol"], "qty": res["qty"],
                "avg_price": res["avg_price"], "txid": res.get("txid")
            })
    except Exception as e:
        write_summary("LIVE BUY ERROR" if not dry_run else "DRY BUY ERROR",
                      {"symbol": pick, "exception": repr(e)})

# -----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        write_summary("FATAL", {"exception": repr(e)})
        raise
