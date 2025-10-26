# trader/adapters.py
# Kraken SELL (and validate) adapter – dependency-free, stdlib only.

from __future__ import annotations
import os, time, json, hmac, hashlib, base64, urllib.parse, urllib.request
from typing import Dict, Any, List, Optional

KRAKEN_API_BASE = "https://api.kraken.com"
API_KEY    = os.environ.get("KRAKEN_API_KEY", "")
API_SECRET = os.environ.get("KRAKEN_API_SECRET", "")

class KrakenError(RuntimeError):
    ...

def _require_keys():
    if not API_KEY or not API_SECRET:
        raise KrakenError("Missing KRAKEN_API_KEY or KRAKEN_API_SECRET")

def _http_json(req: urllib.request.Request, timeout: int = 15) -> Dict[str, Any]:
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def _private(path: str, data: Dict[str, str]) -> Dict[str, Any]:
    _require_keys()
    urlpath = f"/0/private/{path}"
    url     = KRAKEN_API_BASE + urlpath
    data    = {**data, "nonce": str(int(time.time()*1000))}
    post    = urllib.parse.urlencode(data).encode()

    message = (str(data["nonce"]) + urllib.parse.urlencode(data)).encode()
    sha256  = hashlib.sha256(message).digest()
    mac     = hmac.new(base64.b64decode(API_SECRET), urlpath.encode()+sha256, hashlib.sha512)
    sig     = base64.b64encode(mac.digest()).decode()

    req = urllib.request.Request(url, data=post, headers={
        "API-Key": API_KEY, "API-Sign": sig, "User-Agent": "SellGuard/1.0",
    })
    resp = _http_json(req)
    if resp.get("error"):
        raise KrakenError(f"Kraken private {path} error: {resp['error']}")
    return resp["result"]

def _public(path: str, params: Dict[str, str]) -> Dict[str, Any]:
    q = urllib.parse.urlencode(params)
    url = f"{KRAKEN_API_BASE}/0/public/{path}?{q}" if q else f"{KRAKEN_API_BASE}/0/public/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "SellGuard/1.0"})
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
    """
    Resolve Kraken asset pair for e.g. 'ETH' -> {'altname':'ETHUSD', ...}
    Tries USD/USDT candidates; returns first matching asset pair dict.
    """
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

def _format_volume(symbol: str, qty: float, pair_info: Dict[str, Any]) -> str:
    lot_dec = int(pair_info.get("lot_decimals", 8))
    fmt = "{:0." + str(lot_dec) + "f}"
    return fmt.format(qty)

def get_open_positions() -> list:
    """
    Optional helper if your bot wants live balances.
    Returns [{symbol, qty}] for non-stable spot balances.
    """
    _require_keys()
    balances = _private("Balance", {})
    out = []
    for asset_code, qty_str in balances.items():
        try:
            qty = float(qty_str)
        except Exception:
            continue
        if qty <= 0: continue
        sym = _normalize_asset(asset_code)
        if sym in {"USD","USDT","USDC","DAI"}:
            continue
        out.append({"symbol": sym, "qty": qty})
    return out

def place_market_sell(symbol: str, qty: float, reduce_only: bool = False, validate: bool = False) -> Dict[str, Any]:
    """
    Place a MARKET SELL on Kraken. If validate=True, Kraken validates only.
    Returns Kraken 'descr'/'txid' payload.
    """
    _require_keys()
    info = _resolve_pair(symbol)
    pair = info.get("altname") or next(iter(info.values())).get("altname")
    volume = _format_volume(symbol, qty, info)
    data = {"pair": pair, "type": "sell", "ordertype": "market", "volume": volume}
    oflags = []
    if reduce_only: oflags.append("reduce-only")
    if oflags: data["oflags"] = ",".join(oflags)
    if validate: data["validate"] = "true"
    return _private("AddOrder", data)
