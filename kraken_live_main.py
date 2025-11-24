import os
import time
import csv
import hashlib
import hmac
import base64
import urllib.parse
from typing import Tuple, Optional, Dict
import json
from pathlib import Path

import requests


# =============================================================================
#  Kraken — 1-Coin Rotation (Monday Baseline v3 — Rotation with Guards)
#
#  Behavior:
#    * Reads BUY_USD, TP_PCT, SL_PCT, DRY_RUN from env.
#    * Reads momentum_candidates.csv (.state or root) and picks best symbol.
#    * Rotation sweep:
#         - Scans your balances.
#         - For every non-USD asset that is NOT the top candidate:
#             • Skips tiny "dust" positions (< MIN_ROTATION_USD).
#             • If it has a trade history and PnL >= MIN_PNL_TO_KEEP,
#               keeps it as a winner.
#             • Otherwise, SELLs it (DRY_RUN respected).
#    * For the top symbol:
#         - If flat  -> BUY ~BUY_USD worth (respecting cooldown).
#         - If long  -> TP/SL/SPike-fade check, SELL when hit.
#           After a LIVE TP/SL/SPIKE-FADE SELL, it can try to BUY the
#           (possibly updated) top candidate again in the same run,
#           respecting the cooldown timer.
#
#  Default tuning (A1):
#    BUY_USD          = 7.0   – modest position size
#    TP_PCT           = 12.0  – bigger winners
#    SL_PCT           = 1.0   – small loss
#    MIN_ROTATION_USD = 2.0   – ignore tiny dust positions
#    MIN_PNL_TO_KEEP  = 10.0  – keep only strong winners
#
#  Spike / cooldown tuning (B & C):
#    SPIKE_ARM_PCT        = 8.0   – consider it a spike once PnL ≥ 8%
#    SPIKE_FADE_DROP_PCT  = 3.0   – if PnL falls ≥3% from max after arm → SELL
#    COOLDOWN_MINUTES     = 60.0  – after a SELL, avoid rebuying same coin
#
#  Env vars (from YAML):
#    KRAKEN_API_KEY, KRAKEN_API_SECRET
#    BUY_USD, TP_PCT, SL_PCT, DRY_RUN
#    MIN_ROTATION_USD, MIN_PNL_TO_KEEP
#    SPIKE_ARM_PCT, SPIKE_FADE_DROP_PCT, COOLDOWN_MINUTES
# =============================================================================


STATE_DIR = Path(".state")
SPIKE_STATE_PATH = STATE_DIR / "spike_state.json"
COOLDOWN_STATE_PATH = STATE_DIR / "cooldown_state.json"


def ensure_state_dir() -> None:
    try:
        STATE_DIR.mkdir(exist_ok=True)
    except Exception:
        # Non-fatal: just run without extra state if this fails.
        pass


def load_json_state(path: Path, default):
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return default
    return default


def save_json_state(path: Path, data) -> None:
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        # Don't crash trading just because state can't be written.
        pass


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
    trades: dict,
    symbol: str
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
    trades: Dict[str, dict],
    cooldowns: Dict[str, float],
    min_balance: float = 1e-8,
) -> None:
    """
    Rotation sweep with guards:
       - For every non-USD asset with a valid <BASE>/USD pair and non-trivial
        balance:
          * If est_value < MIN_ROTATION_USD      -> skip (dust).
          * If unrealized PnL >= MIN_PNL_TO_KEEP -> keep as winner.
          * Else                                 -> SELL (or DRY-RUN log).
    """
    top_base = top_symbol.split("/")[0].upper()

    # A1 tuning defaults: 2 USD dust, 10% winner threshold
    min_rotation_usd = env_float("MIN_ROTATION_USD", 2.0)
    min_pnl_to_keep = env_float("MIN_PNL_TO_KEEP", 10.0)

    print("\n[ROTATION] Sweeping non-top positions...")
    print(f"[ROTATION] Top candidate this run : {top_symbol}")
    print(f"[ROTATION] MIN_ROTATION_USD       : {min_rotation_usd:.2f} USD")
    print(
        f"[ROTATION] MIN_PNL_TO_KEEP        : {min_pnl_to_keep:.2f}% "
        f"(winners above this are kept)"
    )

    for asset_key, raw_balance in balances.items():
        base = normalize_base_from_balance_key(asset_key)
        if not base:
            continue

        # Skip USD balances
        if base == "USD":
            continue

        # Skip the current top base asset (handled separately)
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

        # Dust filter: don't bother rotating tiny scraps.
        if est_value < min_rotation_usd:
            print(
                f"[ROTATION] Skipping {candidate_symbol}: "
                f"~{est_value:.2f} USD < MIN_ROTATION_USD ({min_rotation_usd:.2f})."
            )
            continue

        # Check PnL for this symbol to decide whether to keep a winner.
        pos_size, avg_entry = infer_position_from_trades(trades, candidate_symbol)
        pnl_pct = None
        if pos_size > 1e-8 and avg_entry:
            pnl_pct = (price - avg_entry) / avg_entry * 100.0
            print(
                f"[ROTATION] {candidate_symbol}: pos={pos_size:.8f}, "
                f"avg_entry={avg_entry:.6f}, PnL={pnl_pct:.2f}%"
            )
            if pnl_pct >= min_pnl_to_keep:
                print(
                    f"[ROTATION] Winner above {min_pnl_to_keep:.2f}% PnL → "
                    f"keeping {candidate_symbol}, no rotation."
                )
                continue

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
            # Start cooldown on this symbol so we don't immediately rebuy it.
            cooldowns[candidate_symbol] = time.time()
        except Exception as e:
            print(
                f"[ERROR] Failed to SELL {candidate_symbol} during rotation sweep: {e}"
            )


def reenter_after_exit(
    api: KrakenTradeAPI,
    previous_symbol: str,
    buy_usd: float,
    dry_run: str,
    cooldowns: Dict[str, float],
    cooldown_minutes: float,
) -> None:
    """
    After a LIVE TP/SL/SPIKE-FADE exit, try to BUY the (possibly updated)
    top candidate again in the same run.

    - Re-reads the momentum CSV (in case the list changed).
    - Uses current USD balance (after the sell).
    - Respects cooldown: won't immediately re-enter a coin we just sold.
    """
    if dry_run == "ON":
        # Should never be called with DRY_RUN=ON, but guard anyway.
        print("[RE-ENTRY] DRY_RUN is ON; not placing a second order.")
        return

    print("\n[RE-ENTRY] Refreshing balances and top candidate for re-entry...")

    # Re-pick in case the candidate list changed.
    new_symbol = pick_symbol_from_csv() or previous_symbol

    # Respect cooldown if we just sold this same symbol.
    now_ts = time.time()
    cd_secs = cooldown_minutes * 60.0
    last_sell = cooldowns.get(new_symbol)
    if last_sell is not None and (now_ts - last_sell) < cd_secs:
        remaining = int(cd_secs - (now_ts - last_sell))
        print(
            f"[RE-ENTRY] Cooldown active for {new_symbol} "
            f"({remaining} seconds left) → staying flat."
        )
        return

    balances = api.get_balance()
    usd_balance = get_usd_balance(balances)
    price = api.get_ticker_price(new_symbol)

    print(f"[RE-ENTRY] Top candidate : {new_symbol}")
    print(f"[RE-ENTRY] USD balance   : {usd_balance:.6f}")
    print(f"[RE-ENTRY] Target BUY_USD: {buy_usd:.2f}")

    # Slight cushion so we don't error out on tiny fee differences.
    if usd_balance < buy_usd * 1.02:
        print("[RE-ENTRY] Not enough USD to re-enter; staying in cash.")
        return

    volume = buy_usd / price
    print(
        f"[RE-ENTRY] LIVE BUY: {volume:.8f} units of {new_symbol} "
        f"(~{buy_usd:.2f} USD at {price:.6f})."
    )

    try:
        result = api.market_buy(new_symbol, volume)
        print("[RE-ENTRY] BUY result:", result)
    except Exception as e:
        print(f"[RE-ENTRY][ERROR] Failed to BUY {new_symbol}: {e}")


# =============================================================================
#  Main rotation logic
# =============================================================================


def main() -> None:
    ensure_state_dir()

    api_key = os.environ.get("KRAKEN_API_KEY", "").strip()
    api_secret = os.environ.get("KRAKEN_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise SystemExit("Missing KRAKEN_API_KEY or KRAKEN_API_SECRET")

    # A1 tuning defaults
    buy_usd = env_float("BUY_USD", 7.0)
    tp_pct = env_float("TP_PCT", 12.0)
    sl_pct = env_float("SL_PCT", 1.0)
    dry_run = os.environ.get("DRY_RUN", "ON").upper()

    # B & C tuning
    spike_arm_pct = env_float("SPIKE_ARM_PCT", 8.0)
    spike_fade_drop_pct = env_float("SPIKE_FADE_DROP_PCT", 3.0)
    cooldown_minutes = env_float("COOLDOWN_MINUTES", 60.0)

    print("============================================================")
    print("  Kraken — 1-Coin Rotation (Monday Baseline v3 — Rotation with Guards)")
    print("------------------------------------------------------------")
    print(f"BUY_USD           : {buy_usd}")
    print(f"TP_PCT            : {tp_pct}")
    print(f"SL_PCT            : {sl_pct}")
    print(f"SPIKE_ARM_PCT     : {spike_arm_pct}")
    print(f"SPIKE_FADE_DROP   : {spike_fade_drop_pct}")
    print(f"COOLDOWN_MINUTES  : {cooldown_minutes}")
    print(f"DRY_RUN           : {dry_run}")
    print("============================================================")

    symbol = pick_symbol_from_csv()
    if not symbol:
        print("No candidate symbol found; nothing to do.")
        return

    api = KrakenTradeAPI(api_key, api_secret)

    # Load persistent spike + cooldown state
    spike_state: Dict[str, Dict[str, float]] = load_json_state(SPIKE_STATE_PATH, {})
    cooldowns: Dict[str, float] = load_json_state(COOLDOWN_STATE_PATH, {})

    # Balances & trades
    balances = api.get_balance()
    usd_balance = get_usd_balance(balances)
    trades_result = api.get_trades_history()
    trades = trades_result.get("trades", {})

    # -------------------------------------------------------------------------
    # 1) ROTATION SWEEP — sell non-top coins (with dust + winner guards)
    # -------------------------------------------------------------------------
    sweep_non_top_positions(api, balances, symbol, dry_run, trades, cooldowns)

    # Save cooldowns potentially updated by sweep
    save_json_state(COOLDOWN_STATE_PATH, cooldowns)

    # -------------------------------------------------------------------------
    # 2) Now handle the top candidate: either BUY if flat or TP/SL/SPIKE-FADE if long
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

    # CASE 1: FLAT -> BUY top candidate (respect cooldown)
    if is_flat:
        print("\n[STATE] Flat in this symbol (after rotation sweep).")

        now_ts = time.time()
        cd_secs = cooldown_minutes * 60.0
        last_sell = cooldowns.get(symbol)
        if last_sell is not None and (now_ts - last_sell) < cd_secs:
            remaining = int(cd_secs - (now_ts - last_sell))
            print(
                f"[COOLDOWN] Recently sold {symbol}; {remaining}s left on cooldown. "
                "Skipping BUY this run."
            )
            return

        if usd_balance < buy_usd * 1.02:
            print("Not enough USD to buy; skipping.")
            return

        volume = buy_usd / price
        print(f"Planned BUY: {volume:.8f} units (~{buy_usd} USD).")

        if dry_run == "ON":
            print("[DRY RUN] Would place MARKET BUY now.")
            return

        print("[LIVE] Sending MARKET BUY...")
        try:
            result = api.market_buy(symbol, volume)
            print("BUY result:", result)
        except Exception as e:
            print(f"[ERROR] Failed to BUY {symbol}: {e}")
        return

    # CASE 2: LONG -> TP/SL/SPIKE-FADE check on the top candidate
    print("\n[STATE] Long position detected for this symbol (top candidate).")
    if not avg_entry:
        print("Cannot compute avg entry; holding position.")
        return

    pnl_pct = (price - avg_entry) / avg_entry * 100.0
    print(f"Unrealized PnL: {pnl_pct:.2f}%")

    # Maintain per-symbol spike max PnL
    sym_state = spike_state.get(symbol, {})
    max_pnl = float(sym_state.get("max_pnl", pnl_pct))
    if pnl_pct > max_pnl:
        max_pnl = pnl_pct
        sym_state["max_pnl"] = max_pnl
        spike_state[symbol] = sym_state
        save_json_state(SPIKE_STATE_PATH, spike_state)

    print(f"[SPIKE] Max PnL seen for {symbol}: {max_pnl:.2f}%")

    take_profit = pnl_pct >= tp_pct
    stop_loss = pnl_pct <= -sl_pct

    # Spike-fade condition: once we've seen a decent spike, if PnL falls off enough, exit.
    spike_armed = max_pnl >= spike_arm_pct
    spike_fade = spike_armed and (pnl_pct <= (max_pnl - spike_fade_drop_pct))

    if not (take_profit or stop_loss or spike_fade):
        print("No TP/SL or spike-fade condition hit; holding.")
        return

    if take_profit:
        reason = "TP"
    elif stop_loss:
        reason = "SL"
    else:
        reason = "SPIKE_FADE"

    print(f"{reason} condition met -> will SELL all of {symbol}.")

    if dry_run == "ON":
        print(f"[DRY RUN] Would SELL {position_size:.8f} units at market.")
        return

    print("[LIVE] Sending MARKET SELL...")
    try:
        result = api.market_sell_all(symbol, position_size)
        print("SELL result:", result)
        # Reset spike state on exit so a new trade starts fresh.
        if symbol in spike_state:
            del spike_state[symbol]
            save_json_state(SPIKE_STATE_PATH, spike_state)
        # Start cooldown from this SELL moment.
        cooldowns[symbol] = time.time()
        save_json_state(COOLDOWN_STATE_PATH, cooldowns)
    except Exception as e:
        print(f"[ERROR] Failed to SELL {symbol}: {e}")
        return

    # After exiting, immediately try to buy the top candidate again
    # in the same RUN (using fresh CSV + balances) if cooldown allows.
    reenter_after_exit(api, symbol, buy_usd, dry_run, cooldowns, cooldown_minutes)


if __name__ == "__main__":
    main()
