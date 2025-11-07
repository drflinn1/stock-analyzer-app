#!/usr/bin/env python3
"""
Main unified runner for Crypto Live workflows.
Adds ALWAYS-ON trailing-stop sell guard (no 30m limitation).

State files it (re)creates in .state/:
- run_summary.json / run_summary.md
- position.json  (single pos: symbol, qty, avg_price, in_position_since, high_price)
- last_ok.txt / last_exit_code.txt
- spike_candidates.csv / momentum_candidates.csv (read-only if present)
"""

import os, json, time, math, subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# ----------------- Paths / State -----------------
STATE_DIR = Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_JSON = STATE_DIR / "run_summary.json"
SUMMARY_MD   = STATE_DIR / "run_summary.md"
LAST_OK      = STATE_DIR / "last_ok.txt"
LAST_EXIT    = STATE_DIR / "last_exit_code.txt"
POS_FILE     = STATE_DIR / "position.json"

CAND_SPIKE   = STATE_DIR / "spike_candidates.csv"
CAND_MOMENT  = STATE_DIR / "momentum_candidates.csv"

# ----------------- Helpers -----------------
def env_str(name: str, default: str = "") -> str:
    val = os.getenv(name, default)
    return "" if val is None else str(val)

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

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())

def log(msg: str) -> None:
    print(msg, flush=True)

def write_file(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)

def read_json(p: Path, default: Any) -> Any:
    try:
        return json.loads(p.read_text())
    except Exception:
        return default

def write_summary_md(data: Dict[str, Any]) -> None:
    # lightweight pretty md
    lines = [
        f"**When:** {data.get('when')}",
        f"**Live (DRY_RUN=OFF):** {data.get('live')}",
        f"**Pick Source:** {data.get('pick_source','')}",
        f"**Pick (symbol):** {data.get('symbol','')}",
        f"**BUY_USD:** {data.get('buy_usd')}",
        f"**Status:** {data.get('status')}",
        f"**Note:** {data.get('note','')}",
        "",
        "### Details",
        "```json",
        json.dumps(data, indent=2),
        "```",
        "",
    ]
    write_file(SUMMARY_MD, "\n".join(lines))

def write_summary_json(data: Dict[str, Any]) -> None:
    write_file(SUMMARY_JSON, json.dumps(data, indent=2))

def ok_exit(code: int = 0):
    write_file(LAST_OK, now_iso())
    write_file(LAST_EXIT, str(code))
    raise SystemExit(code)

# ----------------- Position store -----------------
def load_pos() -> Dict[str, Any]:
    return read_json(POS_FILE, {
        "symbol": None,
        "qty": 0.0,
        "avg_price": 0.0,
        "in_position_since": None,
        "high_price": None,       # <-- trailing high tracked here
        "source": None,
        "buy_usd": 0.0,
        "amount": 0.0,
    })

def save_pos(d: Dict[str, Any]) -> None:
    write_file(POS_FILE, json.dumps(d, indent=2))

# ----------------- Exchange price helpers -----------------
def normalize_pair(symbol: str) -> str:
    """Accepts 'SOON' or 'SOON/USD' or 'SOONUSD' and returns 'SOON/USD'."""
    s = symbol.replace("-", "/").upper().strip()
    if "/" in s:
        base, quote = s.split("/", 1)
        return f"{base}/{quote}"
    if s.endswith("USD"):
        return f"{s[:-3]}/USD"
    return f"{s}/USD"

def read_candidates() -> Optional[str]:
    # prefer momentum candidates, then spike list, else UNIVERSE_PICK
    if CAND_MOMENT.exists():
        try:
            # format: symbol,quote,rank ...
            rows = [r.strip() for r in CAND_MOMENT.read_text().splitlines() if r.strip()]
            if len(rows) >= 2:
                # skip header
                first = rows[1].split(",")[0].strip()
                return normalize_pair(first)
        except Exception:
            pass
    if CAND_SPIKE.exists():
        try:
            rows = [r.strip() for r in CAND_SPIKE.read_text().splitlines() if r.strip()]
            if len(rows) >= 2:
                first = rows[1].split(",")[0].strip()
                return normalize_pair(first)
        except Exception:
            pass
    return None

# ----------------- Always-on trailing stop guard -----------------
def update_trailing_high(current_price: float) -> None:
    pos = load_pos()
    if pos.get("symbol") and current_price > 0:
        hp = pos.get("high_price") or current_price
        if current_price > hp:
            pos["high_price"] = current_price
            save_pos(pos)

def decide_exit(
    symbol: str,
    current_price: float,
    tp_pct: float,
    stop_pct: float,
    slow_min_pct: float,
    window_min: int,
    minutes_held: int
) -> Tuple[bool, str]:
    """
    Returns (should_sell, reason)
    - TP if gain from entry >= tp_pct
    - ALWAYS-ON trailing stop: sell if drop from trailing high >= stop_pct
    - SLOW EXIT: after window_min minutes if gain < slow_min_pct
    """
    pos = load_pos()
    if not pos.get("symbol"):
        return (False, "NO_POS")

    entry = pos.get("avg_price") or 0.0
    hp    = pos.get("high_price") or entry or current_price

    if entry <= 0 or current_price <= 0:
        return (False, "NO_PRICE")

    pnl_from_entry = (current_price / entry - 1.0) * 100.0
    drop_from_high = (1.0 - current_price / (hp if hp > 0 else current_price)) * 100.0

    detail = {
        "symbol": symbol,
        "current": current_price,
        "entry": entry,
        "high": hp,
        "pnl_pct": round(pnl_from_entry, 3),
        "drop_from_high_pct": round(drop_from_high, 3),
        "minutes_held": minutes_held,
    }
    # append to summary json (light)
    write_summary_json({"guard": detail, "when": now_iso()})

    if pnl_from_entry >= tp_pct:
        return (True, f"TP_REACHED_{tp_pct}%")

    if drop_from_high >= stop_pct:
        return (True, f"TRAIL_STOP_{stop_pct}%")

    if window_min and minutes_held >= window_min and pnl_from_entry < slow_min_pct:
        return (True, f"SLOW_EXIT_{slow_min_pct}%/{window_min}m")

    return (False, "HOLD")

def minutes_since(iso_str: Optional[str]) -> int:
    if not iso_str:
        return 0
    try:
        # very small parser; iso like "2025-11-07T02:51:06+00:00"
        ts = time.strptime(iso_str[:19], "%Y-%m-%dT%H:%M:%S")
        then = int(time.mktime(ts))
        return max(0, int(time.time()) - then) // 60
    except Exception:
        return 0

# ----------------- Live price & order shims -----------------
# NOTE:
# These two functions are the only place you may need to align names with
# your existing adapter (if you already had working live trades, tweak here only).
# They expect:
#   - symbol like 'SOON/USD'
#   - qty is base-asset amount (for sell)
#   - usd_amount is the dollar spend (for buy)
#
# They respect DRY_RUN automatically.
#

def get_last_price(symbol: str) -> float:
    """
    Lightweight price getter via Kraken public API.
    Falls back to 0.0 on error.
    """
    import requests
    pair = symbol.replace("/", "")
    try:
        r = requests.get(f"https://api.kraken.com/0/public/Ticker", params={"pair": pair}, timeout=8)
        j = r.json()
        # Kraken returns like {"result":{"SOONUSD":{"c":["2.0506","1"] ... } } }
        res = list(j.get("result", {}).values())
        if not res:
            return 0.0
        last = res[0].get("c", ["0"])[0]
        return float(last)
    except Exception:
        return 0.0

def place_market_buy(symbol: str, usd_amount: float, live: bool) -> Dict[str, Any]:
    """
    Minimal market buy.
    If 'live' is False, simulate only.
    """
    if not live:
        return {"simulated": True, "desc": f"DRY buy {usd_amount} {symbol}"}

    # ---- If you already had a working adapter, replace the block below with your call ----
    # We use the force-sell helper contract (same signing) to send AddOrder via a small Python one-liner
    # to keep this file self-contained.
    try:
        import requests, urllib.parse, hashlib, hmac, base64
        key    = os.getenv("KRAKEN_API_KEY","")
        secret = os.getenv("KRAKEN_API_SECRET","")
        if not key or not secret:
            return {"error":"MISSING_SECRETS"}

        def _private(url_path, data):
            nonce = str(int(time.time()*1000))
            data["nonce"] = nonce
            postdata = urllib.parse.urlencode(data)
            message  = (nonce + postdata).encode()
            sha256   = hashlib.sha256(message).digest()
            mac_data = url_path.encode() + sha256
            mac = hmac.new(base64.b64decode(secret), mac_data, hashlib.sha512)
            headers = {"API-Key": key, "API-Sign": base64.b64encode(mac.digest())}
            r = requests.post("https://api.kraken.com" + url_path, headers=headers, data=data, timeout=10)
            return r.json()

        pair = symbol.replace("/", "")
        resp = _private("/0/private/AddOrder", {
            "ordertype":"market",
            "type":"buy",
            "pair": pair,
            "oflags":"viqc",   # spend in quote currency (USD)
            "cost": str(usd_amount)
        })
        if resp.get("error"):
            return {"error": resp["error"]}
        return {"ok": True, "result": resp.get("result", {})}
    except Exception as e:
        return {"error": str(e)}

def place_market_sell(symbol: str, qty: float, live: bool) -> Dict[str, Any]:
    if not live:
        return {"simulated": True, "desc": f"DRY sell {qty} {symbol}"}
    try:
        import requests, urllib.parse, hashlib, hmac, base64
        key    = os.getenv("KRAKEN_API_KEY","")
        secret = os.getenv("KRAKEN_API_SECRET","")
        if not key or not secret:
            return {"error":"MISSING_SECRETS"}

        def _private(url_path, data):
            nonce = str(int(time.time()*1000))
            data["nonce"] = nonce
            postdata = urllib.parse.urlencode(data)
            message  = (nonce + postdata).encode()
            sha256   = hashlib.sha256(message).digest()
            mac_data = url_path.encode() + sha256
            mac = hmac.new(base64.b64decode(secret), mac_data, hashlib.sha512)
            headers = {"API-Key": key, "API-Sign": base64.b64encode(mac.digest())}
            r = requests.post("https://api.kraken.com" + url_path, headers=headers, data=data, timeout=10)
            return r.json()

        pair = symbol.replace("/", "")
        resp = _private("/0/private/AddOrder", {
            "ordertype":"market",
            "type":"sell",
            "pair": pair,
            "volume": str(qty)
        })
        if resp.get("error"):
            return {"error": resp["error"]}
        return {"ok": True, "result": resp.get("result", {})}
    except Exception as e:
        return {"error": str(e)}

# ----------------- Core logic -----------------
def main():
    # env
    DRY_RUN   = env_str("DRY_RUN", "ON").upper() != "OFF"   # ON => simulate
    BUY_USD   = env_float("BUY_USD", 15.0)
    RESERVE   = env_float("RESERVE_CASH_PCT", 0.0)

    TP_PCT    = env_float("TP_PCT", 5.0)
    STOP_PCT  = env_float("STOP_PCT", 1.0)
    SLOW_MIN  = env_float("SLOW_MIN_PCT", 3.0)
    WINDOW_MIN= env_int("WINDOW_MIN", 30)

    pick_env  = env_str("UNIVERSE_PICK", "").strip().upper()

    live = not DRY_RUN

    # ---- pick symbol ----
    symbol = None
    src    = ""
    if pick_env:
        symbol = normalize_pair(pick_env)
        src    = "UNIVERSE_PICK"
    else:
        cand = read_candidates()
        if cand:
            symbol = cand
            src    = "CANDIDATES_CSV"

    data = {
        "when": now_iso(),
        "live": live,
        "buy_usd": BUY_USD,
        "reserve_cash_pct": RESERVE,
        "universe_pick": pick_env,
        "thresholds": {
            "tp_pct": TP_PCT,
            "stop_pct": STOP_PCT,
            "slow_min_pct": SLOW_MIN,
            "window_min": WINDOW_MIN,
        },
        "status": "INIT",
        "note": "",
        "symbol": symbol or "",
        "pick_source": src,
        "open_position": {},
    }

    # ---- load position ----
    pos = load_pos()
    data["open_position"] = pos

    # ---- price + trailing-high maintenance ----
    if pos.get("symbol"):
        px = get_last_price(pos["symbol"])
        update_trailing_high(px)

        held_min = minutes_since(pos.get("in_position_since"))
        should_sell, reason = decide_exit(
            symbol=pos["symbol"],
            current_price=px,
            tp_pct=TP_PCT,
            stop_pct=STOP_PCT,
            slow_min_pct=SLOW_MIN,
            window_min=WINDOW_MIN,
            minutes_held=held_min
        )
        if should_sell and pos["qty"] > 0:
            sell_res = place_market_sell(pos["symbol"], float(pos["qty"]), live)
            if "error" in sell_res:
                data["status"] = "SELL_ERROR"
                data["note"]   = str(sell_res["error"])
            else:
                data["status"] = "LIVE_SELL_OK" if live else "DRY_SELL_OK"
                data["order"]  = sell_res
                # clear position
                save_pos({"symbol": None, "qty": 0, "avg_price": 0.0, "in_position_since": None, "high_price": None})
        else:
            data["status"] = "HOLD"
            data["note"]   = "Guard=HOLD"

    # ---- buy if flat ----
    pos = load_pos()
    if (not pos.get("symbol")) and symbol:
        px = get_last_price(symbol)
        if px > 0 and BUY_USD > 0:
            # market buy for BUY_USD; record qty approximated afterwards
            buy_res = place_market_buy(symbol, BUY_USD, live)
            if "error" in buy_res:
                data["status"] = "BUY_ERROR"
                data["note"]   = str(buy_res["error"])
            else:
                # record new position; qty approximated by BUY_USD/px
                qty = round(BUY_USD / px, 8)
                new_pos = {
                    "symbol": symbol,
                    "qty": qty,
                    "avg_price": px,
                    "in_position_since": now_iso(),
                    "high_price": px,
                    "source": src,
                    "buy_usd": BUY_USD,
                    "amount": round(BUY_USD/px, 8)
                }
                save_pos(new_pos)
                data["status"] = "LIVE_BUY_OK" if live else "DRY_BUY_OK"
                data["order"]  = buy_res
        else:
            data["status"] = "SKIP_BUY"
            data["note"]   = "No price or BUY_USD=0"

    # ---- write summary ----
    write_summary_json(data)
    write_summary_md(data)
    ok_exit(0)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        write_summary_json({"when": now_iso(), "status": "CRASH", "error": str(e)})
        write_summary_md({"when": now_iso(), "status": "CRASH", "note": str(e)})
        write_file(LAST_EXIT, "1")
        raise
