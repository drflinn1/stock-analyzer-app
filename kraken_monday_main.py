"""
Kraken — 1-Coin Rotation (Monday Baseline v2)

- Checks existing USD-quoted spot position.
- Applies TP/SL to decide sells.
- If flat → rotates into the next fresh-momentum USD pair.
- Skips region-restricted pairs like:
    EAccount:Invalid permissions:PARTI trading restricted for US:WA.
- Supports DRY_RUN ("ON" = simulate, "OFF" = place real orders).

Expected environment variables (set by GitHub Actions YAML):
    BUY_USD   – e.g. "20"
    TP_PCT    – e.g. "8"
    SL_PCT    – e.g. "2"
    DRY_RUN   – "ON" or "OFF"
"""

import os
import time
import traceback
from typing import Dict, Any, List, Optional, Set

# We assume this helper already exists in your repo.
# In earlier steps we based everything on this.
from kraken_client import build_kraken_client_from_env


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    val = os.getenv(name)
    if val is None or not val.strip():
        return default
    return val.strip()


def _get_current_price_from_ohlc(api, pair: str) -> Optional[float]:
    """
    Use 1-minute OHLC candles to approximate the current price.
    Avoids adding new API methods; we already rely on get_ohlc().
    """
    try:
        ohlc = api.get_ohlc(pair, interval=1)
        if not ohlc:
            return None
        return float(ohlc[-1][4])  # close
    except Exception as e:
        print(f"⚠️ Could not fetch current price for {pair}: {e}")
        return None


# ---------------------------------------------------------------------------
# STEP A + B + C: Momentum scan + rotation
# ---------------------------------------------------------------------------

def momentum_scan(api, excluded_pairs: Optional[Set[str]] = None) -> Optional[str]:
    """
    Find the best fresh-momentum USD pair, skipping any in excluded_pairs.

    Logic:
      - Only consider USD pairs.
      - Fresh 5-minute momentum: p_now > p_1m > p_3m > p_5m.
      - Volume spike: last 1m volume > 1.25x avg of last 15m.
      - Uses 1-minute OHLC from api.get_ohlc(pair, interval=1).
    """
    if excluded_pairs is None:
        excluded_pairs = set()

    print("🔍 Momentum scan: early spike + exclusions")

    try:
        tradable = api.list_tradable_pairs()
    except Exception as e:
        print(f"⚠️ Could not list tradable pairs: {e}")
        return None

    usd_pairs = [p for p in tradable if p.endswith("USD")]

    candidates: List[Dict[str, Any]] = []

    for pair in usd_pairs:
        if pair in excluded_pairs:
            continue

        try:
            ohlc = api.get_ohlc(pair, interval=1)  # 1-minute candles
            closes = [float(c[4]) for c in ohlc]
            vols = [float(c[6]) for c in ohlc]

            if len(closes) < 30:
                continue

            # ---- Fresh momentum – last 5 minutes only ----
            p_now = closes[-1]
            p_1m = closes[-2]
            p_3m = closes[-4]
            p_5m = closes[-6]

            fresh_momentum = p_now > p_1m > p_3m > p_5m
            if not fresh_momentum:
                continue  # falling or weak trend

            # ---- Volume check (15m baseline) ----
            last_15_vols = vols[-15:] if len(vols) >= 15 else vols
            if not last_15_vols:
                continue

            avg_vol = sum(last_15_vols) / len(last_15_vols)
            vol_now = vols[-1]
            if avg_vol <= 0:
                continue

            if vol_now < avg_vol * 1.25:
                continue  # no volume confirmation

            # ---- 1-hour context (if enough candles) ----
            if len(closes) >= 60:
                p_60m = closes[-60]
            else:
                p_60m = closes[0]

            pct_60m = (p_now - p_60m) / p_60m * 100 if p_60m > 0 else 0.0

            candidates.append(
                {
                    "pair": pair,
                    "momentum_5m": (p_now - p_5m) / p_5m * 100 if p_5m > 0 else 0.0,
                    "momentum_60m": pct_60m,
                    "vol_boost": vol_now / avg_vol,
                }
            )

        except Exception as e:
            print(f"Scanner error {pair}: {e}")
            continue

    if not candidates:
        print("❌ No valid momentum candidates.")
        return None

    # Sort by strongest fresh 5m momentum first
    best = sorted(candidates, key=lambda x: x["momentum_5m"], reverse=True)[0]
    print(
        f"📈 Selected: {best['pair']} "
        f"(5m +{best['momentum_5m']:.2f}%, "
        f"1h {best['momentum_60m']:.2f}%, "
        f"vol x{best['vol_boost']:.2f})"
    )

    return best["pair"]


def rotate_if_flat(api) -> None:
    """
    Step B + C:
      - If we are flat (no USD-quoted spot position), buy the next momentum coin.
      - Skips any pairs that are restricted for your region and moves on.

    Fixes:
      - "Sell but no buy???"
      - "Invalid permissions: trading restricted for US:WA" → now we skip those.
    """
    print("🔁 Rotation check – are we flat?")

    try:
        positions = api.get_open_spot_positions()
    except Exception as e:
        print(f"⚠️ Could not fetch open positions: {e}")
        traceback.print_exc()
        return

    # Are we already in a USD-quoted position?
    in_position = False
    for pos in positions:
        try:
            pair = str(pos.get("pair") or pos.get("symbol") or "")
            qty = float(pos.get("qty") or pos.get("vol") or 0.0)
        except Exception:
            continue

        if pair.endswith("USD") and qty > 0:
            in_position = True
            print(f"📌 Still in a USD position ({pair}, qty={qty}) → no new buy.")
            break

    if in_position:
        return

    print("✅ Flat (no USD position) → looking for next coin to buy...")

    # How much USD to buy (from env defaults; matches YAML inputs)
    buy_usd = _env_float("BUY_USD", 20.0)
    if buy_usd <= 0:
        print(f"⚠️ BUY_USD={buy_usd} is <= 0 → skipping buy.")
        return

    dry_run_flag = _env_str("DRY_RUN", "ON").upper()
    is_dry_run = dry_run_flag == "ON"
    print(f"💵 BUY_USD={buy_usd:.2f}, DRY_RUN={dry_run_flag}")

    # Skip list for restricted / failing symbols
    blocked: Set[str] = set()
    max_attempts = 5

    for attempt in range(1, max_attempts + 1):
        print(f"🔎 Rotation attempt {attempt}/{max_attempts} (blocked={blocked})")
        pair = momentum_scan(api, excluded_pairs=blocked)

        if not pair:
            print("❌ No more candidates from scanner — giving up for this run.")
            return

        print(f"👉 Trying to BUY {pair} for ~${buy_usd:.2f}")

        try:
            if is_dry_run:
                print(f"🧪 DRY_RUN=ON → Simulated market BUY {pair} for ${buy_usd:.2f}")
                return

            # Assumes your Kraken client has a market_buy(pair, usd_notional) helper.
            order = api.market_buy(pair, buy_usd)
            print(f"✅ Buy order placed: {order}")
            return  # done, we got in

        except Exception as e:
            msg = str(e)
            print(f"⚠️ Buy failed for {pair}: {msg}")

            # Step C: auto-skip region-restricted / permission-denied pairs
            restricted_keywords = [
                "Invalid permissions",
                "trading restricted",
                "EAccount:",
                "EOrder:PermissionDenied",
            ]

            if any(k in msg for k in restricted_keywords):
                print(f"⛔ {pair} appears restricted; adding to blocked list and retrying.")
                blocked.add(pair)
                continue  # try next candidate

            # Any other error → don't loop endlessly; just stop
            print("❌ Non-permission error; not retrying further this run.")
            return

    print("❌ Reached max rotation attempts with no successful buy.")


# ---------------------------------------------------------------------------
# TP/SL sell logic
# ---------------------------------------------------------------------------

def check_tp_sl_and_maybe_sell(api, tp_pct: float, sl_pct: float, is_dry_run: bool) -> None:
    """
    Walk open USD-quoted spot positions and apply TP/SL.

    For each position:
      - Infer entry price: cost / vol
      - Get current price from 1m OHLC
      - If PnL% >= TP_PCT → SELL
      - If PnL% <= -SL_PCT → SELL
    """
    print(f"🧮 Checking TP/SL on open positions (TP={tp_pct}%, SL={sl_pct}%)")

    try:
        positions = api.get_open_spot_positions()
    except Exception as e:
        print(f"⚠️ Could not fetch open positions for TP/SL: {e}")
        traceback.print_exc()
        return

    for pos in positions:
        try:
            pair = str(pos.get("pair") or pos.get("symbol") or "")
            qty = float(pos.get("qty") or pos.get("vol") or 0.0)
            cost = float(pos.get("cost") or pos.get("cost_basis") or 0.0)
        except Exception:
            continue

        if not pair.endswith("USD") or qty <= 0 or cost <= 0:
            continue

        entry_price = cost / qty
        current_price = _get_current_price_from_ohlc(api, pair)
        if current_price is None or entry_price <= 0:
            continue

        pnl_pct = (current_price - entry_price) / entry_price * 100.0

        print(
            f"🔎 {pair}: entry={entry_price:.6f}, "
            f"current={current_price:.6f}, PnL={pnl_pct:.2f}%"
        )

        should_sell = False
        reason = ""

        if pnl_pct >= tp_pct:
            should_sell = True
            reason = f"TP hit ({pnl_pct:.2f}% ≥ {tp_pct}%)"
        elif pnl_pct <= -sl_pct:
            should_sell = True
            reason = f"SL hit ({pnl_pct:.2f}% ≤ -{sl_pct}%)"

        if not should_sell:
            continue

        print(f"🚨 Sell trigger for {pair} ({reason}), qty={qty}")

        try:
            if is_dry_run:
                print(
                    f"🧪 DRY_RUN=ON → Simulated market SELL {pair}, "
                    f"qty={qty} (reason: {reason})"
                )
            else:
                order = api.market_sell(pair, qty)
                print(f"✅ Sell order placed for {pair}: {order}")
        except Exception as e:
            print(f"❌ Failed to sell {pair}: {e}")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("🚀 Kraken — 1-Coin Rotation (Monday Baseline v2)")
    print("   Starting run at:", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    dry_run_flag = _env_str("DRY_RUN", "ON").upper()
    is_dry_run = dry_run_flag == "ON"

    tp_pct = _env_float("TP_PCT", 8.0)
    sl_pct = _env_float("SL_PCT", 2.0)

    print(f"Settings: DRY_RUN={dry_run_flag}, TP_PCT={tp_pct}, SL_PCT={sl_pct}")

    api = build_kraken_client_from_env()

    # 1) First, enforce TP/SL on any existing USD-quoted position
    check_tp_sl_and_maybe_sell(api, tp_pct=tp_pct, sl_pct=sl_pct, is_dry_run=is_dry_run)

    # 2) Then, if we are flat, rotate into the next coin
    rotate_if_flat(api)

    print("✅ Run complete.")


if __name__ == "__main__":
    main()
