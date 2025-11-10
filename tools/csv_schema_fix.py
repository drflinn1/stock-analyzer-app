#!/usr/bin/env python3
"""
Normalize candidate CSVs so rotation always sees the columns it expects.
Independent version (no imports from trader/).
"""

import csv
import math
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

STATE_DIR = Path(".state")
FILES = [STATE_DIR / "momentum_candidates.csv", STATE_DIR / "spike_candidates.csv"]
KRAKEN_URL = "https://api.kraken.com/0/public/Ticker"


def normalize_pair(symbol: str) -> str:
    """Normalize crypto pair names like 'SOLUSD' or 'SOL-USD' -> 'SOL/USD'."""
    if not symbol:
        return ""
    s = symbol.strip().upper().replace("-", "/")
    if not s.endswith("/USD") and "USD" in s and "/" not in s:
        s = s.replace("USD", "/USD")
    return s


def get_public_quote(pair: str) -> Optional[float]:
    """Get current price from Kraken public API."""
    try:
        sym = pair.replace("/", "")
        resp = requests.get(KRAKEN_URL, params={"pair": sym}, timeout=8)
        data = resp.json().get("result", {})
        if not data:
            return None
        ticker = next(iter(data.values()))
        price = float(ticker["c"][0])
        return price
    except Exception:
        return None


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
    scored, has_any = [], False
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
        rd = csv.DictReader(f)
        headers = [h.strip() for h in (rd.fieldnames or [])]
        for row in rd:
            rows.append({k.strip(): v for k, v in row.items()})
    if not rows:
        return

    # pair
    for r in rows:
        p = r.get("pair") or r.get("symbol") or r.get("PAIR") or r.get("SYMBOL") or ""
        r["pair"] = normalize_pair(str(p))

    # quote
    for r in rows:
        q = _to_float(r.get("quote"))
        if q is None:
            q = _to_float(get_public_quote(r["pair"]))
        r["quote"] = f"{q:.12f}" if q is not None else ""

    # rank
    ranks = [_to_int(r.get("rank")) for r in rows]
    if any(rv is None for rv in ranks):
        auto = _rank_from_scores(rows, "score")
        for i, r in enumerate(rows):
            r["rank"] = ranks[i] if ranks[i] is not None else auto[i]
    else:
        for r in rows:
            r["rank"] = _to_int(r["rank"])

    # headers = core first then extras
    core = ["pair", "rank", "quote"]
    seen, extras = set(core), []
    for h in headers:
        if h.lower() in ("pair", "rank", "quote"):
            continue
        if h not in seen:
            extras.append(h)
            seen.add(h)
    out_headers = core + extras

    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=out_headers)
        wr.writeheader()
        for r in rows:
            out = {h: r.get(h, "") for h in extras}
            out["pair"] = r.get("pair", "")
            out["rank"] = r.get("rank", "")
            out["quote"] = r.get("quote", "")
            wr.writerow(out)
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
