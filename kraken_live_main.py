import os
import time
import hashlib
import hmac
import base64
import urllib.parse
from typing import Tuple, Optional

import requests


# =============================================================================
#  Minimal Kraken trading + LIVE 1-coin TP/SL logic
#
#  Env variables (set by your GitHub Actions workflow):
#    KRAKEN_API_KEY      - from Kraken
#    KRAKEN_API_SECRET   - from Kraken (base64 string)
#    BUY_USD             - USD notional to buy when flat (e.g. "5")
#    TP_PCT              - Take-profit % (e.g. "8" for 8%)
#    SL_PCT              - Stop-loss % (e.g. "1" for -1%)
#    DRY_RUN             - "ON" to simulate, "OFF" to place real orders
#
#  Optional:
#    SYMBOL              - trading pair, default "SOL/USD"
# =============================================================================


class KrakenTradeAPI:
    """Minimal Kraken trading wrapper for spot market buy/sell + helpers."""

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = base64.b64decode(api_secret)
        self.base_url = "https://api.kraken.com"

    # ---------- low-level signing / POST ----------

    def _sign(self, urlpath: str, data: dict) -> str:
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data["nonce"]) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        signature = hmac.new(self.api_secret, message, hashlib.sha512)
        return base64.b64encode(signature.digest()).decode()

    def _post(self, path: str, data: dict) -> dict:
        url = self.base_url + path
        headers = {
            "API-Key": self.api_key,
            "API-Sign": self._sign(path, data),
        }
        resp = requests.post(url, headers=headers, data=data, timeout=20)
        resp.raise_for_status()
        js = resp.json()
        if js.get("error"):
            raise RuntimeError(f"Kraken error: {js['error']}")
        return js["result"]

    # ---------- public helpers ----------

    def get_ticker_price(self, symbol: str) -> float:
        """Return last traded price for the given symbol, e.g. 'SOL/USD'."""
        pair = symbol.replace("/", "").upper()
        url = f"{self.base_url}/0/public/Ticker"
        resp = requests.get(url, params={"pair": pair}, timeout=20)
        resp.raise_for_status()
        js = resp.json()
        if js.get("error"):
            raise RuntimeError(f"Kraken public error: {js['error']}")
        result = js["result"]
        # result key may be "SOLUSD" or a prefixed variant, just take first
        first_key = next(iter(result.keys()))
        price_str = result[first_key]["c"][0]
        return float(price_str)

    def get_balance(self) -> dict:
        """Return balances dict from /Balance."""
        data = {"nonce": int(time.time() * 1000)}
        return self._post("/0/private/Balance", data)

    def get_trades_history(self) -> dict:
        """Return full trades history dict from /TradesHistory."""
        data = {"nonce": int(time.time() * 1000)}
        return self._post("/0/private/TradesHistory", data)

    # ---------- order helpers ----------

    def _market_order(self, symbol: str, side: str, volume: str) -> dict:
        pair = symbol.replace("/", "").upper()
        data = {
            "nonce": int(time.time() * 1000),
            "ordertype": "market",
            "type": side,
            "volume": volume,
            "pair": pair,
        }
        return self._post("/0/private/AddOrder", data)

    def market_buy(self, symbol: str, volume: float) -> dict:
        """Buy volume units of base asset at market."""
        return self._market_order(symbol, "buy", f"{volume:.8f}")

    def market_sell_all(self, symbol: str, volume: float) -> dict:
        """Sell volume units of base asset at market."""
        return self._market_order(symbol, "sell", f"{volume:.8f}")


# =============================================================================
#  Position inference from trade history
# =============================================================================


def infer_position_from_trades(
    trades: dict, symbol: str
) -> Tuple[float, Optional[float]]:
    """
    Compute net position size (base units) and average entry price
    from full trade history for the given symbol.

    Returns:
        (net_position, avg_entry_price) where:
          - net_position > 0 => long that many units
          - avg_entry_price is None if no position
    """
    pair_code = symbol.replace("/", "").upper()
    net_vol = 0.0
    net_cost = 0.0

    # trades is a dict: { txid: { ... } }
    for txid, t in trades.items():
        pair = str(t.get("pair", ""))
        if pair_code not in pair:
            continue

        vol = float(t.get("vol", 0.0))
        cost = float(t.get("cost", 0.0))
        side = t.get("type")

        if side == "buy":
            net_vol += vol
            net_cost += cost
        elif side == "sell":
            net_vol -= vol
            net_cost -= cost

    if net_vol > 0:
        avg_entry = net_cost / net_vol if net_cost != 0 else None
        return net_vol, avg_entry

    return 0.0, None


def get_asset_balance(balances: dict, asset: str) -> float:
    """
    Try multiple Kraken-style asset codes to get a balance for `asset`,
    e.g. asset='SOL' -> check 'SOL', 'XSOL', 'ZSOL', etc.
    """
    candidates = [
        asset,
        asset.upper(),
        "X" + asset.upper(),
        "Z" + asset.upper(),
    ]
    for key in candidates:
        if key in balances:
            try:
                return float(balances[key])
            except ValueError:
                continue
    return 0.0


def get_usd_balance(balances: dict) -> float:
    for key in ("ZUSD", "USD"):
        if key in balances:
            try:
                return float(balances[key])
            except ValueError:
                continue
    return 0.0


# =============================================================================
#  Main LIVE routine
# =============================================================================


def main() -> None:
    # ---- read env vars ----
    api_key = os.environ.get("KRAKEN_API_KEY", "").strip()
    api_secret = os.environ.get("KRAKEN_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise SystemExit("Missing KRAKEN_API_KEY or KRAKEN_API_SECRET env vars")

    symbol = os.environ.get("SYMBOL", "SOL/USD").strip()

    def env_float(name: str, default: float) -> float:
        val = os.environ.get(name)
        if not val:
            return default
        try:
            return float(val)
        except ValueError:
            return default

    buy_usd = env_float("BUY_USD", 5.0)
    tp_pct = env_float("TP_PCT", 8.0)
    sl_pct = env_float("SL_PCT", 1.0)
    dry_run = os.environ.get("DRY_RUN", "ON").upper()

    print("============================================================")
    print("   Kraken LIVE 1-Coin Bot")
    print("------------------------------------------------------------")
    print(f"Symbol      : {symbol}")
    print(f"BUY_USD     : {buy_usd}")
    print(f"TP_PCT      : {tp_pct}")
    print(f"SL_PCT      : {sl_pct}")
    print(f"DRY_RUN     : {dry_run}")
    print("============================================================")

    api = KrakenTradeAPI(api_key, api_secret)

    # ---- fetch balances & trades ----
    balances = api.get_balance()
    usd_balance = get_usd_balance(balances)

    base_asset = symbol.split("/")[0].upper()
    base_balance = get_asset_balance(balances, base_asset)

    trades_result = api.get_trades_history()
    trades = trades_result.get("trades", {})
    net_pos, avg_entry = infer_position_from_trades(trades, symbol)

    # Use trade-based position if available, else fallback to balance
    position_size = net_pos if net_pos > 0 else base_balance

    # ---- fetch current price ----
    price = api.get_ticker_price(symbol)

    print(f"Current price for {symbol}: {price:.6f} USD")
    print(f"USD balance                : {usd_balance:.6f}")
    print(f"{base_asset} balance           : {base_balance:.8f}")
    print(f"Inferred open position     : {position_size:.8f}")
    if avg_entry:
        print(f"Average entry price        : {avg_entry:.6f} USD")

    # ---- decide action ----
    is_flat = position_size <= 1e-8

    if is_flat:
        print("\n[STATE] Flat (no open position).")
        if usd_balance < buy_usd * 1.02:
            print("Not enough USD to buy, skipping.")
            return

        volume = buy_usd / price
        print(f"Planned BUY volume: {volume:.8f} {base_asset} (~{buy_usd} USD).")

        if dry_run == "ON":
            print("[DRY RUN] Would place MARKET BUY now.")
            return

        print("[LIVE] Sending MARKET BUY...")
        result = api.market_buy(symbol, volume)
        print("BUY order result:")
        print(result)
        return

    # ---- we are long, check TP/SL ----
    print("\n[STATE] Long position detected.")
    if not avg_entry:
        print("Could not determine average entry price from history.")
        print("Holding position (no TP/SL check).")
        return

    pnl_pct = (price - avg_entry) / avg_entry * 100.0
    print(f"Unrealized PnL: {pnl_pct:.2f}%")

    should_take_profit = pnl_pct >= tp_pct
    should_stop_loss = pnl_pct <= -sl_pct

    if not (should_take_profit or should_stop_loss):
        print("Neither TP nor SL hit. Holding position.")
        return

    reason = "TP" if should_take_profit else "SL"
    print(f"{reason} condition met → will SELL all {base_asset}.")

    if dry_run == "ON":
        print(f"[DRY RUN] Would place MARKET SELL of {position_size:.8f} {base_asset}.")
        return

    print("[LIVE] Sending MARKET SELL...")
    result = api.market_sell_all(symbol, position_size)
    print("SELL order result:")
    print(result)


if __name__ == "__main__":
    main()
