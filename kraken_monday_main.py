import os
import time
import csv
import hashlib
import hmac
import base64
import urllib.parse
from typing import Tuple, Optional, Dict

import requests


# =============================================================================
#  Kraken — 1-Coin Rotation (Monday Baseline v3 — Pure Rotation, C1)
#
#  Behavior:
#    * Reads BUY_USD, TP_PCT, SL_PCT, DRY_RUN from env.
#    * Reads momentum_candidates.csv (.state or root) and picks best symbol.
#    * NEW: Scans your balances and SELLS every non-USD asset that is NOT
#           the current top candidate (pure 1-coin rotation).
#    * For the top symbol:
#         - If flat  -> BUY ~BUY_USD worth (DRY_RUN respected).
#         - If long  -> TP/SL check, SELL when hit.
#
#  Env vars (from YAML):
#    KRAKEN_API_KEY, KRAKEN_API_SECRET
#    BUY_USD, TP_PCT, SL_PCT, DRY_RUN
# =============================================================================


class KrakenTradeAPI:
    """Minimal Kraken wrapper for spot trading + basic queries."""

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
        """
        Get the last trade price for a symbol like 'SOL/USD'.
        """
        pair = symbol.replace("/", "").upper()
        url = f"{self.base_url}/0/public/Ticker"
        resp = requests.get(url, params={"pair": pair}, timeout=20)
        resp.raise_for_status()
        js = resp.json()
        if js.get("error"):
            raise RuntimeError(f"Kraken public error: {js['error']}")
        result = js["result"]
        first_key = next(iter(result.keys()))
        price_str = result[first_key]["c"][0]
        return float(price_str)

    def get_balance(self) -> dict:
        data = {"nonce": int(time.time() * 1000)}
        return self._post("/0/private/Balance", data)

    def get_trades_history(self) -> dict:
        data = {"nonce": int(time.time() * 1000)}
        return self._post("/0/private/TradesHistory", data)

    # ---------- order helpers ----------

    def _market_order(self, symbol: str, side: str, volume: float) -> dict:
        pair = symbol.replace("/", "").upper()
        data = {
            "nonce": int(time.time() * 1000),
            "ordertype": "market",
            "type": side,
            "volume": f"{volume:.8f}",
            "pair": pair,
        }
        return self._post("/0/private/AddOrder", data)

    def market_buy(self, symbol: str, volume: float) -> dict:
        return self._market_order(symbol, "buy", volume)

    def market_sell_all(self, symbol: str, volume: float) -> dict:
        return self._market_order(symbol, "sell", volume)


# =============================================================================
#  Helpers: position inference, balances, CSV selection, rotation sweep
# =============================================================================


def infer_position_from_trades(
    trades: dict, symbol: str
) -> Tuple[float, Optional[float]]:
    """
    Return (net_position_units, avg_entry_price) for this symbol.

    We match on a loose basis:
      - symbol 'LSK/USD' -> pair_code 'LSKUSD'
      - trade 'pair' fields like 'LSKUSD' or 'XLSKZUSD' will both match.
    """
    pair_code = symbol.replace("/", "").upper()
    net_vol = 0.0
    net_cost = 0.0

    for txid, t in trades.items():
        pair = str(t.get("pair", ""))
        if pair_code not in pair:
            continue

        try:
            vol = float(t.get("vol", 0.0))
            cost = float(t.get("cost", 0.0))
        except (TypeError, ValueError):
            continue

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


def get_usd_balance(balances: dict) -> float:
    for key in ("ZUSD", "USD"):
        if key in balances:
            try:
                return float(balances[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def pick_symbol_from_csv() -> Optional[str]:
    """
    Pick the best symbol (lowest rank) from momentum_candidates CSV.

    We support both:
      - .state/momentum_candidates.csv  (what the scan currently writes)
      - momentum_candidates.csv         (older location, or manual export)
    """
    candidate_paths = [
        ".state/momentum_candidates.csv",
        "momentum_candidates.csv",
    ]

    csv_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            csv_path = path
            break

    if not csv_path:
        print(f"[WARN] No momentum_candidates CSV found in {candidate_paths}")
        return None

    print(f"[INFO] Using momentum candidates from: {csv_path}")

    best_symbol = None
    best_rank = None

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row.get("symbol")
            rank_str = row.get("rank")
            if not symbol or rank_str is None:
                continue
            try:
                rank = float(rank_str)
            except ValueError:
                continue

            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_symbol = symbol

    if best_symbol:
        print(f"[INFO] Picked top candidate from CSV: {best_symbol} (rank={best_rank})")
    else:
        print("[WARN] No valid rows in momentum_candidates CSV")

    return best_symbol


def env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def normalize_base_from_balance_key(key: str) -> Optional[str]:
    """
    Convert Kraken balance keys to a base asset code.

    Examples:
      'ZUSD'  -> 'USD'
      'USD'   -> 'USD'
      'XXBT'  -> 'XBT'
      'XTRX'  -> 'TRX'
      'SOL'   -> 'SOL'
    """
    if not key:
        return None

    # USD special-cased
    if key.upper() in ("USD", "ZUSD"):
        return "USD"

    k = key.upper()

    # Many crypto assets are prefixed with X or Z (e.g. XXBT, XTRX, ZETH)
    if k.startswith(("X", "Z")) and len(k) > 3:
        k = k[1:]  # strip one leading char -> XXBT -> XBT, XTRX -> TRX

    return k


def sweep_non_top_positions(
    api: KrakenTradeAPI,
    balances: Dict[str, str],
    top_symbol: str,
    dry_run: str,
    min_balance: float = 1e-8,
) -> None:
    """
    PURE ROTATION (C1):
      - For every non-USD asset with a valid <BASE>/USD pair
        and non-trivial balance:
          * If it's NOT the top_symbol, SELL it.

    Example:
      top_symbol = 'MIRA/USD'
      balances -> LSK, TRX, SOL, ACT, etc.
      => we will try to SELL all of those except MIRA.
    """
    top_base = top_symbol.split("/")[0].upper()
    print("\n[ROTATION] Sweeping non-top positions...")
    print(f"[ROTATION] Top candidate this run: {top_symbol}")

    for asset_key, raw_balance in balances.items():
        base = normalize_base_from_balance_key(asset_key)
        if not base:
            continue

        # Skip USD balances
        if base == "USD":
            continue

        # Skip the current top base asset (we'll handle it separately)
        if base == top_base:
            continue

        try:
            bal = float(raw_balance)
        except (TypeError, ValueError):
            continue

        if bal <= min_balance:
            continue

        candidate_symbol = f"{base}/USD"

        # Verify that this asset has a valid USD pair on Kraken.
        try:
            price = api.get_ticker_price(candidate_symbol)
        except Exception:
            # No valid pair or some issue; ignore this asset.
            continue

        est_value = bal * price
        print(
            f"[ROTATION] Found non-top position: {asset_key} "
            f"({candidate_symbol}) bal={bal:.8f} (~{est_value:.2f} USD)"
        )

        if dry_run == "ON":
            print(
                f"[DRY RUN] Would SELL {bal:.8f} {base} at market "
                f"({candidate_symbol}) as part of rotation."
            )
            continue

        print(
            f"[LIVE] Rotating out of {base} ({candidate_symbol}) "
            f"by selling {bal:.8f} units at market..."
        )
        try:
            result = api.market_sell_all(candidate_symbol, bal)
            print(f"[LIVE] SELL result for {candidate_symbol}: {result}")
        except Exception as e:
            print(
                f"[ERROR] Failed to SELL {candidate_symbol} during rotation sweep: {e}"
            )


# =============================================================================
#  Main rotation logic
# =============================================================================


def main() -> None:
    api_key = os.environ.get("KRAKEN_API_KEY", "").strip()
    api_secret = os.environ.get("KRAKEN_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise SystemExit("Missing KRAKEN_API_KEY or KRAKEN_API_SECRET")

    buy_usd = env_float("BUY_USD", 20.0)
    tp_pct = env_float("TP_PCT", 8.0)
    sl_pct = env_float("SL_PCT", 1.0)
    dry_run = os.environ.get("DRY_RUN", "ON").upper()

    print("============================================================")
    print("  Kraken — 1-Coin Rotation (Monday Baseline v3 — Pure Rotation, C1)")
    print("------------------------------------------------------------")
    print(f"BUY_USD : {buy_usd}")
    print(f"TP_PCT  : {tp_pct}")
    print(f"SL_PCT  : {sl_pct}")
    print(f"DRY_RUN : {dry_run}")
    print("============================================================")

    symbol = pick_symbol_from_csv()
    if not symbol:
        print("No candidate symbol found; nothing to do.")
        return

    api = KrakenTradeAPI(api_key, api_secret)

    # Balances & trades
    balances = api.get_balance()
    usd_balance = get_usd_balance(balances)
    trades_result = api.get_trades_history()
    trades = trades_result.get("trades", {})

    # -------------------------------------------------------------------------
    # 1) PURE ROTATION SWEEP — sell everything that isn't the top candidate
    # -------------------------------------------------------------------------
    sweep_non_top_positions(api, balances, symbol, dry_run)

    # -------------------------------------------------------------------------
    # 2) Now handle the top candidate: either BUY if flat or TP/SL if long
    # -------------------------------------------------------------------------
    position_size, avg_entry = infer_position_from_trades(trades, symbol)
    price = api.get_ticker_price(symbol)

    print(f"\nSelected symbol       : {symbol}")
    print(f"Current price         : {price:.6f} USD")
    print(f"USD balance           : {usd_balance:.6f}")
    print(f"Inferred position     : {position_size:.8f} units")
    if avg_entry:
        print(f"Average entry price   : {avg_entry:.6f} USD")

    is_flat = position_size <= 1e-8

    # CASE 1: FLAT -> BUY top candidate
    if is_flat:
        print("\n[STATE] Flat in this symbol (after rotation sweep).")

        if usd_balance < buy_usd * 1.02:
            print("Not enough USD to buy; skipping.")
            return

        volume = buy_usd / price
        print(f"Planned BUY: {volume:.8f} units (~{buy_usd} USD).")

        if dry_run == "ON":
            print("[DRY RUN] Would place MARKET BUY now.")
            return

        print("[LIVE] Sending MARKET BUY...")
        result = api.market_buy(symbol, volume)
        print("BUY result:", result)
        return

    # CASE 2: LONG -> TP/SL check on the top candidate
    print("\n[STATE] Long position detected for this symbol (top candidate).")
    if not avg_entry:
        print("Cannot compute avg entry; holding position.")
        return

    pnl_pct = (price - avg_entry) / avg_entry * 100.0
    print(f"Unrealized PnL: {pnl_pct:.2f}%")

    take_profit = pnl_pct >= tp_pct
    stop_loss = pnl_pct <= -sl_pct

    if not (take_profit or stop_loss):
        print("Neither TP nor SL hit; holding.")
        return

    reason = "TP" if take_profit else "SL"
    print(f"{reason} condition met -> will SELL all of {symbol}.")

    if dry_run == "ON":
        print(f"[DRY RUN] Would SELL {position_size:.8f} units at market.")
        return

    print("[LIVE] Sending MARKET SELL...")
    result = api.market_sell_all(symbol, position_size)
    print("SELL result:", result)


if __name__ == "__main__":
    main()
