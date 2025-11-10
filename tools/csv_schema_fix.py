#!/usr/bin/env python3
"""
Normalize candidate CSVs so rotation always sees the columns it expects.

Targets (if present):
  .state/momentum_candidates.csv
  .state/spike_candidates.csv

Guarantees each row has:
  - pair  (normalized like 'SOL/USD')
  - rank  (int; higher is better; from existing rank or derived from 'score')
  - quote (float; filled from Kraken public price if missing)

Other columns (score, pct24, usd_vol, ema_slope, etc.) are preserved.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

from trader.crypto_engine import get_public_quote, normalize_pair

STATE_DIR = Path(".state")
FILES = [STATE_DIR / "momentum_candidates.csv", STATE_DIR / "spike_candidates.csv"]


def _to_float(x: Any) -> Optional[float]:
    try:
        v = float(str(x).strip())
        return v if math.isfinite(v) and v > 0 else None
    except Exception:
        return None


def _to_int(x: Any) -> Optional[int]:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None


def _rank_from_scores(rows: List[Dict[str, Any]], score_key: str = "score") -> List[int]:
    """Highest score -> highest rank number (100, 99, ...)."""
    scored = []
    has_any = False
    for i, r in enumerate(rows):
        try:
            val = float(r.get(score_key))
            has_any = True
        except Exception:
            val = float("-inf")
        scored.append((i, val))
    if has_any:
        scored.sort(key=lambda t: t[1], reverse=True)
        base = 100
        rank_map = {idx: base - j for j, (idx, _) in enumerate(scored)}
        return [rank_map[i] for i in range(len(rows))]
    return [100 - i for i in range(len(rows))]


def _normalize_file(path: Path) -> None:
    if not path.exists():
        return

    rows: List[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in (reader.fieldnames or [])]
        for row in reader:
            rows.append({k.strip(): v for k, v in row.items()})

    if not rows:
        return

    # Normalize pair
    for r in rows:
        p = r.get("pair") or r.get("symbol") or r.get("PAIR") or r.get("SYMBOL") or ""
        r["pair"] = normalize_pair(str(p))

    # Ensure quote
    for r in rows:
        q = _to_float(r.get("quote"))
        if q is None:
            q = _to_float(get_public_quote(r["pair"]))
        r["quote"] = f"{q:.12f}" if q is not None else ""

    # Ensure rank
    ranks = [_to_int(r.get("rank")) for r in rows]
    if any(rv is None for rv in ranks):
        auto = _rank_from_scores(rows, "score")
        for i, r in enumerate(rows):
            r["rank"] = ranks[i] if ranks[i] is not None else auto[i]
    else:
        for r in rows:
            r["rank"] = _to_int(r["rank"])

    # Core headers first, then original extras (no dups)
    core = ["pair", "rank", "quote"]
    extras = []
    seen = set(core)
    for h in headers:
        if h.lower() in ("pair", "rank", "quote"):
            continue
        if h not in seen:
            extras.append(h)
            seen.add(h)
    out_headers = core + extras

    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_headers)
        w.writeheader()
        for r in rows:
            out = {h: r.get(h, "") for h in extras}
            out["pair"] = r.get("pair", "")
            out["rank"] = r.get("rank", "")
            out["quote"] = r.get("quote", "")
            w.writerow(out)
    tmp.replace(path)


def main() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    for p in FILES:
        try:
            _normalize_file(p)
        except Exception as e:
            (STATE_DIR / "csv_fix_error.log").write_text(f"{p.name}: {e}\n", encoding="utf-8")
    (STATE_DIR / "last_csv_fix.txt").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
