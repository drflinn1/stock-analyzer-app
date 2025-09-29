#!/usr/bin/env python3
import os, time, math, csv
from datetime import datetime
from typing import Dict, Tuple, Optional

import ccxt

USD_KEYS = {"USD", "ZUSD"}
STABLES = {"USDT", "USDC", "DAI"}  # will be converted to USD too

def now_ts():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def main():
    dry_run = (os.getenv("DRY_RUN", "true").lower() == "true")
    api_key = os.getenv("KRAKEN_API_KEY", "")
    api_sec = os.getenv("KRAKEN_API_SECRET", "")
    api_pwd = os.getenv("KRAKEN_API_PASSWORD") or None

    print("="*80)
    print(f"[{now_ts()}] 🚨 FORCE LIQUIDATE START  (DRY_RUN={dry_run})")
    print("="*80)

    if not api_key or not api_sec:
        print("ERROR: Missing KRAKEN_API_KEY or KRAKEN_API_SECRET.")
        raise SystemExit(2)

    exchange = ccxt.kraken({
        "apiKey": api_key,
        "secret": api_sec,
        "password": api_pwd,
        "enableRateLimit": True,
        # Uncomment if needed:
        # "options": {"warnOnFetchOpenOrdersWithoutSymbol": False},
    })

    exchange.load_markets()
    markets: Dict[str, dict] = exchange.markets

    # 0) Cancel all open orders
    try:
        print(f"[{now_ts()}] Cancelling open orders…")
        open_orders = exchange.fetch_open_orders()
        if open_orders:
            print(f"Found {len(open_orders)} open orders.")
            if not dry_run:
                for o in open_orders:
                    try:
                        exchange.cancel_order(o["id"], symbol=o.get("symbol"))
                        time.sleep(exchange.rateLimit/1000)
                    except Exception as e:
                        print(f"Cancel failed for {o.get('id')}: {e}")
            else:
                print("(DRY RUN) Skipping actual cancels.")
        else:
            print("No open orders.")
    except Exception as e:
        print(f"Warn: could not fetch/cancel open orders: {e}")

    # 1) Snapshot balances → CSV
    print(f"[{now_ts()}] Fetching balances…")
    bal = exchange.fetch_balance()
    total = bal.get("total", {})
    snapshot_path = f"force_liquidate_snapshot_{int(time.time())}.csv"
    with open(snapshot_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["asset", "total"])
        for k, v in sorted(total.items()):
            amt = safe_float(v)
            if amt:
                w.writerow([k, amt])
    print(f"Saved pre-liquidation snapshot: {snapshot_path}")

    # 2) Build liquidation list (everything except USD/ZUSD)
    liquidation: Dict[str, float] = {}
    usd_total = 0.0
    for asset, amount in total.items():
        amt = safe_float(amount)
        if amt <= 0: 
            continue
        if asset in USD_KEYS:
            usd_total += amt
            continue
        liquidation[asset] = amt

    print(f"USD/ZUSD balance: {usd_total:.2f}")
    if not liquidation:
        print("Nothing to liquidate. Exiting cleanly.")
        return

    # Helper: find symbol to sell asset into USD. Fallback to USDT → USD path.
    def pick_symbol_to_usd(base: str) -> Tuple[Optional[str], Optional[str]]:
        # Prefer direct BASE/USD
        direct = f"{base}/USD"
        if direct in markets and markets[direct]["active"]:
            return direct, None
        # try lowercase variations (ccxt often normalizes, but be safe)
        direct2 = f"{base.upper()}/USD"
        if direct2 in markets and markets[direct2]["active"]:
            return direct2, None

        # Next: BASE/USDT then USDT/USD
        b_usdt = f"{base}/USDT"
        if b_usdt in markets and markets[b_usdt]["active"]:
            if "USDT/USD" in markets and markets["USDT/USD"]["active"]:
                return b_usdt, "USDT/USD"
        b_usdt2 = f"{base.upper()}/USDT"
        if b_usdt2 in markets and markets[b_usdt2]["active"]:
            if "USDT/USD" in markets and markets["USDT/USD"]["active"]:
                return b_usdt2, "USDT/USD"

        # Next: BASE/USDC then USDC/USD
        b_usdc = f"{base}/USDC"
        if b_usdc in markets and markets[b_usdc]["active"]:
            if "USDC/USD" in markets and markets["USDC/USD"]["active"]:
                return b_usdc, "USDC/USD"
        b_usdc2 = f"{base.upper()}/USDC"
        if b_usdc2 in markets and markets[b_usdc2]["active"]:
            if "USDC/USD" in markets and markets["USDC/USD"]["active"]:
                return b_usdc2, "USDC/USD"

        return None, None

    # 3) Sell each asset → USD
    sell_results = []
    for asset, amt in liquidation.items():
        base = asset.upper()
        # Kraken reports some assets with prefixes (e.g., XETH, XXBT) but ccxt usually normalizes.
        # Try stripping a leading 'X' or 'Z' if symbol not found:
        candidates = [base]
        if base.startswith(("X","Z")) and len(base) > 3:
            candidates.append(base[1:])
        if base.endswith(".S"):  # staking tokens etc. sellable?
            candidates.append(base.replace(".S",""))
        selected = None

        for candidate in candidates:
            sym1, sym2 = pick_symbol_to_usd(candidate)
            if sym1:
                selected = (candidate, sym1, sym2)
                break

        print("-"*80)
        print(f"[{now_ts()}] Asset {asset} — amount {amt}")
        if not selected:
            print(f"!! No USD path found for {asset}. Skipping (left as-is).")
            continue

        candidate, first_leg, second_leg = selected
        # First leg: sell BASE into quote (USD/USDT/USDC)
        try:
            m1 = markets[first_leg]
            # Fetch min amount
            min_amt = m1.get("limits", {}).get("amount", {}).get("min") or 0.0
            precision = m1.get("precision", {}).get("amount", 8)
            use_amt = max(0.0, float(amt) * 0.999)  # leave tiny dust
            if use_amt < min_amt:
                print(f"Too small for {first_leg}: amt={use_amt} < min={min_amt}. Skipping.")
                continue
            use_amt = float(f"{use_amt:.{precision}f}")

            print(f"First leg: MARKET SELL {use_amt} {candidate} on {first_leg}")
            if not dry_run:
                order1 = exchange.create_order(first_leg, "market", "sell", use_amt)
                print(f"  → Order1 id {order1.get('id')}")
            else:
                print("  (DRY RUN) Not placing order.")
            time.sleep(exchange.rateLimit/1000)
        except Exception as e:
            print(f"ERROR first leg {first_leg}: {e}")
            continue

        # Second leg (if any): convert stable → USD
        if second_leg:
            try:
                m2 = markets[second_leg]
                qprecision = m2.get("precision", {}).get("amount", 8)

                # Get updated balance for that stable
                time.sleep(exchange.rateLimit/1000)
                bal2 = exchange.fetch_balance() if not dry_run else {"total": {second_leg.split("/")[0]: 0}}
                stable_ccy = second_leg.split("/")[0]
                stable_amt = safe_float(bal2.get("total", {}).get(stable_ccy, amt)) * 0.999
                if stable_amt <= 0:
                    print(f"Second leg: no {stable_ccy} detected (may be pending). You can re-run later.")
                else:
                    stable_amt = float(f"{stable_amt:.{qprecision}f}")
                    print(f"Second leg: MARKET SELL {stable_amt} {stable_ccy} on {second_leg}")
                    if not dry_run:
                        order2 = exchange.create_order(second_leg, "market", "sell", stable_amt)
                        print(f"  → Order2 id {order2.get('id')}")
                    else:
                        print("  (DRY RUN) Not placing order.")
                    time.sleep(exchange.rateLimit/1000)
            except Exception as e:
                print(f"ERROR second leg {second_leg}: {e}")

        sell_results.append((asset, amt, first_leg, second_leg or ""))

    # 4) Final USD total
    try:
        time.sleep(exchange.rateLimit/1000)
        final_bal = exchange.fetch_balance()
        final_usd = sum(safe_float(final_bal.get("total", {}).get(k, 0)) for k in USD_KEYS)
        print("-"*80)
        print(f"[{now_ts()}] ✅ Done. USD-equivalents after liquidations: {final_usd:.2f} (DRY_RUN={dry_run})")
    except Exception as e:
        print(f"Warn: could not fetch final USD: {e}")

    print("="*80)
    print("SUMMARY of attempted sells:")
    for row in sell_results:
        print(f"  - {row[0]}  amt={row[1]}  via {row[2]}  then {row[3]}")
    print("="*80)

if __name__ == "__main__":
    main()
