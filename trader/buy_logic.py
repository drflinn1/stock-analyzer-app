# trader/buy_logic.py
# Logic to place BUY orders on Kraken via the adapted API.

from __future__ import annotations
import os, json, time, urllib.parse, urllib.request
from typing import Any, Dict, List

KRAKEN_API_BASE = "https://api.kraken.com"
API_KEY    = os.environ.get("KRAKEN_API_KEY", "")
API_SECRET = os.environ.get("KRAKEN_API_SECRET", "")

class KrakenError(RuntimeError):
    pass

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
    data    = {**data, "nonce": str(int(time.time() * 1000))}
    post    = urllib.parse.urlencode(data).encode()

    message = (str(data["nonce"]) + urllib.parse.urlencode(data)).encode()
    sha256  = hashlib.sha256(message).digest()
    mac     = hmac.new(base64.b64decode(API_SECRET), urlpath.encode() + sha256, hashlib.sha512)
    sig     = base64.b64encode(mac.digest()).decode()

    req = urllib.request.Request(url, data=post, headers={
        "API-Key": API_KEY, "API-Sign": sig, "User-Agent": "BuyGuard/1.0",
    })
    resp = _http_json(req)
    if resp.get("error"):
        raise KrakenError(f"Kraken private {path} error: {resp['error']}")
    return resp["result"]

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
        alt = v.get("altname", "").upper()
        if alt.endswith("USD"):
            best = v; break
        if best is None:
            best = v
    if not best:
        raise KrakenError(f"No tradable pair found for {symbol}")
    return best

def _normalize_asset(a: str) -> str:
    a = a.replace(".S", "").replace(".M", "")
    if len(a) >= 2 and a[0] in "XZ": a = a[1:]
    if len(a) >= 2 and a[0] in "XZ": a = a[1:]
    return a.upper()

def place_market_buy(symbol: str, qty: float) -> Dict[str, Any]:
    """
    Place a MARKET BUY on Kraken.
    """
    _require_keys()
    info = _resolve_pair(symbol)
    pair = info.get("altname") or next(iter(info.values())).get("altname")
    volume = _format_volume(symbol, qty, info)
    data = {"pair": pair, "type": "buy", "ordertype": "market", "volume": volume}

    return _private("AddOrder", data)

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
        if sym in {"USD", "USDT", "USDC", "DAI"}:
            continue
        out.append({"symbol": sym, "qty": qty})
    return out

def get_buy_decision(symbol: str, qty: float, last_price: float, min_buy_usd: float = 10) -> str:
    """
    This function makes the decision if a buy should be placed based on the criteria you define.
    Example criteria might include the USD value of the purchase, etc.
    """
    usd_value = qty * last_price
    if usd_value < min_buy_usd:
        return "SKIP_DUST"
    return "BUY"

def process_buy(symbol: str, qty: float, last_price: float) -> Dict[str, Any]:
    """
    Handles buying logic and execution. Decides whether to buy based on criteria.
    """
    decision = get_buy_decision(symbol, qty, last_price)
    if decision == "SKIP_DUST":
        return {"status": "SKIP_DUST", "symbol": symbol, "qty": qty, "reason": "below minimum USD threshold"}
    
    try:
        resp = place_market_buy(symbol, qty)
        return {"status": "BUY_ORDER_PLACED", "response": resp}
    except Exception as e:
        return {"status": "BUY_FAILED", "error": str(e)}

