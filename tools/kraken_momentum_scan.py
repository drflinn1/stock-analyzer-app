# tools/kraken_momentum_scan.py
#
# Kraken – Global USD Gainers Scan
#
# - Scans ALL Kraken spot markets quoted in USD.
# - Uses 24h change (last vs open) to rank gainers.
# - Applies basic liquidity filters so we don't chase totally illiquid junk.
# - Writes results to .state/momentum_candidates.csv for the rotation bot.
#
# Output CSV columns:
#   symbol        -> e.g. "LSK/USD"
#   pair_code     -> e.g. "LSKUSD" (Kraken altname)
#   last_price    -> last trade price
#   pct_change_24h-> 24h % change
#   vol_base_24h  -> 24h volume in base units
#   vol_usd_24h   -> 24h notional volume in USD
#   rank          -> 1 = strongest gainer
#
# This is designed to be called from GitHub Actions, but you can also run it
# locally:  python tools/kraken_momentum_scan.py

import csv
import os
import sys
from typing import Dict, List, Tuple

import requests

BASE_URL = "https://api.kraken.com"


def fetch_asset_pairs() -> Dict[str, dict]:
    """Fetch all asset pairs from Kraken."""
    url = f"{BASE_URL}/0/public/AssetPairs"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    js = resp.json()
    if js.get("error"):
        raise RuntimeError(f"Kraken AssetPairs error: {js['error']}")
    return js["result"]


def fetch_ticker_for_pairs(pair_codes: List[str]) -> Dict[str, dict]:
    """Fetch ticker info for a list of altname pair codes (e.g. ['LSKUSD', 'XBTUSD'])."""
    result: Dict[str, dict] = {}

    # Kraken supports multiple pairs in one call via comma-separated 'pair='
    # but we'll chunk it to avoid URL length issues.
    CHUNK_SIZE = 20
    for i in range(0, len(pair_codes), CHUNK_SIZE):
        chunk = pair_codes[i : i + CHUNK_SIZE]
        if not chunk:
            continue

        params = {"pair": ",".join(chunk)}
        url = f"{BASE_URL}/0/public/Ticker"

        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            js = resp.json()
        except Exception as e:
            print(f"[WARN] Ticker request failed for chunk {chunk}: {e}", file=sys.stderr)
            continue

        if js.get("error"):
            print(f"[WARN] Kraken Ticker error for chunk {chunk}: {js['error']}", file=sys.stderr)
            continue

        # Merge into result dict
        for k, v in js.get("result", {}).items():
            result[k] = v

    return result


def build_usd_universe() -> List[Tuple[str, str, dict]]:
    """
    Build a list of (symbol, altname, pair_meta) for all USD-quoted spot pairs.

    symbol:   "BASE/USD" for our bot (e.g. "LSK/USD")
    altname:  Kraken altname pair code (e.g. "LSKUSD")
    pair_meta: raw AssetPairs metadata dict
    """
    pairs = fetch_asset_pairs()
    universe: List[Tuple[str, str, dict]] = []

    for pair_key, meta in pairs.items():
        # Skip dark / internal pairs
        if meta.get("darkpool", False):
            continue

        altname = meta.get("altname", "")
        base = meta.get("base", "")
        quote = meta.get("quote", "")

        # We want things quoted in USD. Kraken usually has:
        #   quote="ZUSD", altname="SOMEUSD"
        # We'll key off altname ending in "USD" & quote containing "USD".
        if not altname.endswith("USD"):
            continue
        if "USD" not in quote.upper():
            continue

        # Normalize base name for our "BASE/USD" symbol
        # Many bases look like "XLSK" / "ZETH" etc.; strip leading X/Z when present.
        base_norm = base.upper()
        if base_norm.startswith(("X", "Z")) and len(base_norm) > 3:
            base_norm = base_norm[1:]

        symbol = f"{base_norm}/USD"
        universe.append((symbol, altname, meta))

    print(f"[INFO] Found {len(universe)} USD-quoted spot pairs in AssetPairs.")
    return universe


def compute_gainers() -> List[dict]:
    """
    Build ranked list of gainers with basic liquidity filters.
    Returns list of dict rows ready to write to CSV.
    """
    universe = build_usd_universe()
    if not universe:
        print("[WARN] No USD-quoted pairs found; returning empty gainers list.")
        return []

    # Map altname -> (symbol, meta)
    alt_to_symbol: Dict[str, Tuple[str, dict]] = {
        altname: (symbol, meta) for (symbol, altname, meta) in universe
    }

    altnames = list(alt_to_symbol.keys())
    ticker = fetch_ticker_for_pairs(altnames)
    if not ticker:
        print("[WARN] No ticker data returned; returning empty gainers list.")
        return []

    rows: List[dict] = []

    # Liquidity thresholds (tweak as needed)
    MIN_24H_NOTIONAL_USD = 10_000.0  # require at least this much 24h traded
    MIN_PRICE = 0.0000001            # avoid zero / absurd
    MAX_PRICE = 100_000.0

    for altname, ticker_data in ticker.items():
        symbol, _meta = alt_to_symbol.get(altname, (None, None))
        if not symbol:
            # Some pairs may come back with internal keys; skip unknowns
            continue

        try:
            last = float(ticker_data["c"][0])    # last trade price
            open_24h = float(ticker_data["o"])   # 24h open price
            vol_24h_base = float(ticker_data["v"][1])  # 24h volume in base units
        except Exception:
            continue

        if not (MIN_PRICE < last < MAX_PRICE):
            continue

        # Compute 24h % change (protect against zero open)
        if open_24h <= 0:
            continue
        pct_change = (last - open_24h) / open_24h * 100.0

        # Notional 24h volume in USD
        vol_24h_usd = vol_24h_base * last

        # Apply liquidity filter
        if vol_24h_usd < MIN_24H_NOTIONAL_USD:
            continue

        rows.append(
            {
                "symbol": symbol,
                "pair_code": altname,
                "last_price": last,
                "pct_change_24h": pct_change,
                "vol_base_24h": vol_24h_base,
                "vol_usd_24h": vol_24h_usd,
            }
        )

    # Rank by pct_change_24h descending
    rows.sort(key=lambda r: r["pct_change_24h"], reverse=True)

    # Add rank column
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    print(f"[INFO] After filters, {len(rows)} USD gainers remain.")
    return rows


def write_csv(rows: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = [
        "symbol",
        "pair_code",
        "last_price",
        "pct_change_24h",
        "vol_base_24h",
        "vol_usd_24h",
        "rank",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[INFO] Wrote {len(rows)} rows to {path}")


def main() -> None:
    try:
        rows = compute_gainers()
    except Exception as e:
        print(f"[ERROR] Failed to compute gainers: {e}", file=sys.stderr)
        sys.exit(1)

    # Even if empty, create the file so the bot logs what happened
    out_path = ".state/momentum_candidates.csv"
    write_csv(rows, out_path)

    if not rows:
        print("[WARN] No gainers passed filters; rotation bot may stay in USD or keep current position.")
    else:
        top = rows[0]
        print(
            "[INFO] Top gainer:",
            top["symbol"],
            f"{top['pct_change_24h']:.2f}%",
            f"(24h vol ≈ {top['vol_usd_24h']:.0f} USD)",
        )


if __name__ == "__main__":
    main()
