# trader/adapters.py
# Kraken BUY/SELL adapters (validate-aware). No 3rd-party deps.

from __future__ import annotations
import os, time, json, hmac, hashlib, base64, urllib.parse, urllib.request
from typing import Dict, Any

KRAKEN_API_BASE = "https://api.kraken.com"
_API_KEY    = os.environ.get("KRAKEN_API_KEY", "")
_API_SECRET = os.environ.get("KRAKEN_API_SECRET", "")

class KrakenError(RuntimeError):
    ...

def _require_keys():
    if not _API_KEY or not _API_SECRET:
        raise KrakenError("Missing KRAKEN_API_KEY or KRAKEN_API_SECRET")

def _http_json(req: urllib.request.Request, timeout: int = 15) -> Dict[str, Any]:
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def _private(path: str, data: Dict[str, str]) -> Dict[str, Any]:
    _require_keys()
    urlpath = f"/0/private/{path}"
    url     = KRAKEN_API_BASE + urlpath
    data    = {**data, "nonce": str(int(time.time() * 1000))}
    post    = urllib.parse.urlencode(data).encode()

    message = (str(data["nonce"]) + urllib.parse.urlencode(data)).encode()
    sha256  = hashlib.sha256(message).digest()
    mac     = hmac.new(base64.b64decode(_API_SECRET), urlpath.encode() + sha256, hashlib.sha512)
    sig     = base64.b64encode(mac.digest()).decode()

    req = urllib.request.Request(url, data=post, headers={
        "API-Key": _API_KEY, "API-Sign": sig, "User-Agent": "TradeBot/1.0",
    })
    resp = _http_json(req)
    if resp.get("error"):
        raise KrakenError(f"Kraken private {path} error: {resp['error']}")
    return resp["result"]

def _public(path: str, params: Dict[str, str]) -> Dict[str, Any]:
    q = urllib.parse.urlencode(params)
    url = f"{KRAKEN_API_BASE}/0/public/{path}?{q}" if q else f"{KRAKEN_API_BASE}/0/public/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "TradeBot/1.0"})
    resp = _http_json(req)
    if resp.get("error"):
        raise KrakenError(f"Kraken public {path} error: {resp['error']}")
    return resp["result"]

def _normalize_asset(a: str) -> str:
    a = a.replace(".S","").replace(".M","")
    if len(a) >= 2 and a[0] in "XZ": a = a[1:]
    if len(a) >= 2 and a[0] in "XZ": a = a[1:]
    return a.upper()

def _resolve_pair(symbol: str) -> Dict[str, Any]:
    s = symbol.upper()
    candidates = ",".join([f"{s}USD", f"{s}USDT", f"X{s}ZUSD", f"{s}USD.P"])
    res = _public("AssetPairs", {"pair": candidates})
    best = None
    for _, v in res.items():
        alt = v.get("altname","").upper()
        if alt.endswith("USD"):
            best = v; break
        if best is None:
            best = v
    if not best:
        raise KrakenError(f"No tradable pair found for {symbol}")
    return best

def _format_volume(qty: float, lot_decimals: int) -> str:
    fmt = "{:0." + str(int(lot_decimals)) + "f}"
    return fmt.format(qty)

def get_usd_balance() -> float:
    """Returns spot USD cash balance (not USDT)."""
    res = _private("Balance", {})
    for k, v in res.items():
        if _normalize_asset(k) == "USD":
            try: return float(v)
            except: return 0.0
    return 0.0

def place_market_sell(symbol: str, qty: float, *, validate: bool = False, reduce_only: bool = False) -> Dict[str, Any]:
    info = _resolve_pair(symbol)
    pair = info.get("altname")
    lot  = int(info.get("lot_decimals", 8))
    data = {
        "pair": pair, "type": "sell", "ordertype": "market",
        "volume": _format_volume(qty, lot)
    }
    if reduce_only: data["oflags"] = "reduce-only"
    if validate:    data["validate"] = "true"
    return _private("AddOrder", data)

def place_market_buy(symbol: str, qty: float, *, validate: bool = False) -> Dict[str, Any]:
    info = _resolve_pair(symbol)
    pair = info.get("altname")
    lot  = int(info.get("lot_decimals", 8))
    data = {
        "pair": pair, "type": "buy", "ordertype": "market",
        "volume": _format_volume(qty, lot)
    }
    if validate: data["validate"] = "true"
    return _private("AddOrder", data)

def last_price_usd(symbol: str) -> float:
    """Best-effort last price in USD (tries USD/USDT pairs)."""
    s = symbol.upper()
    pairs = ",".join([f"{s}USD", f"{s}USDT", f"X{s}ZUSD", f"{s}USD.P"])
    res = _public("Ticker", {"pair": pairs})
    # Prefer *USD if available
    for k, v in res.items():
        if str(k).upper().endswith("USD"):
            return float(v["c"][0])
    # otherwise take first
    k0 = next(iter(res))
    return float(res[k0]["c"][0])
