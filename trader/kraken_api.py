import time
import requests
import hashlib
import hmac
import base64
import urllib.parse
from decimal import Decimal


class KrakenTradeAPI:
    """
    Thin Kraken trading wrapper.
    Works for market buy/sell using USD pairs.
    """

    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = base64.b64decode(api_secret)
        self.base_url = "https://api.kraken.com"

    def _sign(self, urlpath, data):
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        signature = hmac.new(self.api_secret, message, hashlib.sha512)
        sigdigest = base64.b64encode(signature.digest())
        return sigdigest.decode()

    def _post(self, path, data):
        url = self.base_url + path
        headers = {
            'API-Key': self.api_key,
            'API-Sign': self._sign(path, data)
        }
        r = requests.post(url, headers=headers, data=data, timeout=10)
        r.raise_for_status()
        return r.json()

    # ------------------------------
    # MARKET BUY with USD notional
    # ------------------------------
    def market_buy_usd(self, pair, usd_amount):
        data = {
            "nonce": int(time.time() * 1000),
            "ordertype": "market",
            "type": "buy",
            "volume": usd_amount,       # let Kraken handle exact qty
            "pair": pair.replace("/", "")
        }
        return self._post("/0/private/AddOrder", data)

    # ------------------------------
    # MARKET SELL full position
    # ------------------------------
    def market_sell_all(self, pair, volume):
        data = {
            "nonce": int(time.time() * 1000),
            "ordertype": "market",
            "type": "sell",
            "volume": volume,
            "pair": pair.replace("/", "")
        }
        return self._post("/0/private/AddOrder", data)
