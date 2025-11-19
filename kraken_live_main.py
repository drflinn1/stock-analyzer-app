import time
import hashlib
import hmac
import base64
import urllib.parse

import requests


class KrakenTradeAPI:
    """Minimal Kraken trading wrapper for market buy/sell."""

    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = base64.b64decode(api_secret)
        self.base_url = "https://api.kraken.com"

    def _sign(self, urlpath, data):
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data["nonce"]) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        signature = hmac.new(self.api_secret, message, hashlib.sha512)
        return base64.b64encode(signature.digest()).decode()

    def _post(self, path, data):
        url = self.base_url + path
        headers = {
            "API-Key": self.api_key,
            "API-Sign": self._sign(path, data),
        }
        resp = requests.post(url, headers=headers, data=data, timeout=15)
        resp.raise_for_status()
        js = resp.json()
        if js.get("error"):
            raise RuntimeError(f"Kraken error: {js['error']}")
        return js

    def _market_order(self, symbol: str, side: str, volume: str):
        pair = symbol.replace("/", "").upper()
        data = {
            "nonce": int(time.time() * 1000),
            "ordertype": "market",
            "type": side,
            "volume": volume,
            "pair": pair,
        }
        return self._post("/0/private/AddOrder", data)

    def market_buy_usd(self, symbol: str, volume: str):
        # volume is base-asset size (e.g. 0.01 BTC)
        return self._market_order(symbol, "buy", volume)

    def market_sell_all(self, symbol: str, volume: str):
        return self._market_order(symbol, "sell", volume)
