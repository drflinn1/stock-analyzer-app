#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/csv_schema_fix.py  (pure Python, no external deps)

Normalize a CSV to the schema: ['symbol', 'quote', 'rank'].

- Accepts many source column names and maps them:
    * symbol: 'symbol', 'pair', 'Pair', 'ticker', 'asset'
    * quote : 'quote', 'price', 'last', 'close'
    * rank  : 'rank'  (if absent, assign 1..N in current order)
- If the input CSV is missing or empty, create an empty _fixed.csv with header.
- Writes a sibling file with suffix '_fixed.csv'.

Usage:
    python tools/csv_schema_fix.py <path_to_csv>
If no path is given, defaults to .state/momentum_candidates.csv
"""

import os
import sys
import csv

DEFAULT_PATH = ".state/momentum_candidates.csv"
TARGET_HEADER = ["symbol", "quote", "rank"]


def ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def write_fixed(path: str, rows):
    """Write rows (list of dicts with symbol/quote/rank) to <path>_fixed.csv."""
    fixed_path = path.replace(".csv", "_fixed.csv")
    ensure_dir(fixed_path)
    with open(fixed_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TARGET_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({
                "symbol": r.get("symbol"),
                "quote": r.get("quote"),
                "rank": r.get("rank"),
            })
    print(f"✅ Normalized schema → {fixed_path}")


def detect_col(header, candidates):
    """Return the first existing column (original case) matching any candidate (case-insensitive)."""
    lower_map = {h.lower(): h for h in header}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def to_float_or_str(val):
    """Try to coerce to float string; if not possible, return as-is (or empty)."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    # strip thousands separators if any
    s2 = s.replace(",", "")
    try:
        # keep as string but normalized to numeric when possible
        return str(float(s2))
    except Exception:
        return s  # leave original (e.g., if it's not numeric)


def normalize_csv(path: str) -> None:
    # If input file doesn't exist, create an empty fixed file and bail.
    if not os.path.exists(path):
        ensure_dir(path)
        write_fixed(path, [])
        return

    # Read input CSV
    with open(path, "r", newline="", encoding="utf-8") as f:
        try:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            rows_in = list(reader) if header else []
        except Exception:
            # If unreadable, emit empty fixed with header
            write_fixed(path, [])
            return

    if not header or not rows_in:
        write_fixed(path, [])
        return

    # Detect source columns
    sym_col = detect_col(header, ["symbol", "pair", "Pair", "ticker", "asset"])
    quo_col = detect_col(header, ["quote", "price", "last", "close"])
    rnk_col = detect_col(header, ["rank"])

    rows_out = []
    next_rank = 1
    for row in rows_in:
        symbol = (row.get(sym_col) if sym_col else None)
        quote = to_float_or_str(row.get(quo_col)) if quo_col else None

        # Rank: keep existing integer if present; else assign sequentially
        if rnk_col and str(row.get(rnk_col, "")).strip():
            try:
                rank_val = int(float(str(row.get(rnk_col)).strip()))
            except Exception:
                rank_val = next_rank
                next_rank += 1
        else:
            rank_val = next_rank
            next_rank += 1

        rows_out.append({
            "symbol": symbol,
            "quote": quote,
            "rank": rank_val,
        })

    write_fixed(path, rows_out)


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    normalize_csv(in_path)
