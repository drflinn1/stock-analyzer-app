# tools/momentum_spike.py
# -----------------------------------------------------------------------------
# Momentum Spike Scanner (SCAN-ONLY, SAFE MODE)
#
# What it does
#   • Scans Kraken symbols quoted in USD/USDT/USDC
#   • Ranks by 24h % change and filters by min 24h % and min USD volume
#   • (Lightweight) EMA check on shortlisted symbols to avoid pure wicks
#   • Saves results to .state/spike_candidates.csv and prints a summary
#
# What it does NOT do
#   • It does NOT place orders. This module is scan-only by design.
#
# Env/Vars you can set (all optional; safe defaults provided)
#   MIN_24H_PCT           default: 25        (trigger threshold)
#   MIN_BASE_VOL_USD      default: 25000     (liquidity floor)
#   EMA_WINDOW            default: 20        (bars on 15m)
#   MAX_RESULTS           default: 10        (how many to keep)
#   TOP_K_PRECHECK        default: 40        (how many symbols get EMA check)
#   EXCLUDE_STABLES       default: true      (filter stablecoin bases)
#   EXCLUDE_LEVERAGED     default: true      (filter tokens like 3L/3S, UP/DOWN)
#   QUOTES                default: USD,USDT,USDC
#
# Output
#   .state/spike_candidates.csv with columns:
#     symbol,last,pct_24h,vol_usd_24h,ema_window,last_close,ema,above_ema,ts
#
# Dependencies
#   pip install ccxt pandas
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import sys
import time
import math
import json
import pathlib
from typing import Dict, Any, List, Tuple

# Graceful import if ccxt/pandas aren't present (the workflow should install them)
try:
    import ccxt  # type: ignore
except Exception as e:
    ccxt = None  # noqa

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # noqa


# ------------------------------ helpers -------------------------------------


def _get_env(name: str, default: str) -> str:
    val = os.getenv(name)
    return str(val).strip() if val not in (None, "") else str(default)


def _get_bool(name: str, default: bool) -> bool:
    raw = _get_env(name, "true" if default else "false").lower()
    return raw in ("1", "true", "yes", "y", "on")


def _get_int(name: str, default: int) -> int:
    try:
        return int(float(_get_env(name, str(default))))
    except Exception:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(_get_env(name, str(default)))
    except Exception:
        return default


def _ensure_state_dir() -> pathlib.Path:
    p = pathlib.Path(".state")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_symbol(symbol: str) -> bool:
    """Quick sanity: exclude extremely long or odd symbols."""
    return 2 <= len(symbol) <= 25 and "/" in symbol


# ------------------------------ config --------------------------------------


MIN_24H_PCT = _get_float("MIN_24H_PCT", 25.0)
MIN_BASE_VOL_USD = _get_float("MIN_BASE_VOL_USD", 25_000.0)
EMA_WINDOW = _get_int("EMA_WINDOW", 20)
MAX_RESULTS = _get_int("MAX_RESULTS", 10)
TOP_K_PRECHECK = _get_int("TOP_K_PRECHECK", 40)
EXCLUDE_STABLES = _get_bool("EXCLUDE_STABLES", True)
EXCLUDE_LEVERAGED = _get_bool("EXCLUDE_LEVERAGED", True)

# Comma list of acceptable quotes
QUOTES = [q.strip().upper() for q in _get_env("QUOTES", "USD,USDT,USDC").split(",") if q.strip()]

STABLE_BASES = {
    "USDT", "USDC", "DAI", "EUR", "GBP", "USD", "BUSD", "TUSD", "FDUSD", "PYUSD"
}
LEV_TOK_PATTERNS = ("3L", "3S", "5L", "5S", "UP", "DOWN", "BULL", "BEAR")

TIMEFRAME = "15m"   # EMA timeframe
EMA_MIN_BARS = max(EMA_WINDOW + 2, 30)  # fetch enough bars for a stable EMA


# ------------------------------ core logic ----------------------------------


def _build_exchange():
    if ccxt is None:
        raise RuntimeError(
            "ccxt is not installed. Make sure your workflow installs 'ccxt'."
        )
    ex = ccxt.kraken({"enableRateLimit": True})
    return ex


def _is_lev_token(symbol: str) -> bool:
    base = symbol.split("/")[0].upper()
    return any(tag in base for tag in LEV_TOK_PATTERNS)


def _want_symbol(symbol: str) -> bool:
    if not _safe_symbol(symbol):
        return False
    base, quote = symbol.split("/")
    quote = quote.upper()
    base = base.upper()

    if quote not in QUOTES:
        return False
    if EXCLUDE_STABLES and base in STABLE_BASES:
        return False
    if EXCLUDE_LEVERAGED and _is_lev_token(symbol):
        return False
    return True


def _usd_volume_from_ticker(tkr: Dict[str, Any], quote: str) -> float:
    """
    Try to derive 24h USD (or quote-USD) volume.
    Priority: quoteVolume if quote ~ USD; else baseVolume*last as approximation.
    """
    last = float(tkr.get("last") or tkr.get("close") or 0.0)
    base_vol = float(tkr.get("baseVolume") or 0.0)
    quote_vol = float(tkr.get("quoteVolume") or 0.0)

    if quote in ("USD", "USDT", "USDC") and quote_vol:
        return float(quote_vol)

    if base_vol and last:
        return float(base_vol * last)

    # Fallback if nothing is available
    return 0.0


def _pct_from_ticker(tkr: Dict[str, Any]) -> float:
    pct = tkr.get("percentage")
    if pct is None:
        # derive percentage from change/last if present
        change = tkr.get("change")
        last = tkr.get("last") or tkr.get("close")
        if change is not None and last:
            try:
                pct = (float(change) / float(last)) * 100.0
            except Exception:
                pct = None
    return float(pct) if pct is not None else float("nan")


def _ema(series: List[float], window: int) -> float:
    if pd is None:
        # lightweight EMA if pandas not present
        k = 2 / (window + 1)
        ema_val = series[0]
        for x in series[1:]:
            ema_val = x * k + ema_val * (1 - k)
        return float(ema_val)
    s = pd.Series(series, dtype=float)
    return float(s.ewm(span=window, adjust=False).mean().iloc[-1])


def _fetch_ohlcv_close(ex, symbol: str, timeframe: str, limit: int) -> List[float]:
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv:
            return []
        closes = [float(x[4]) for x in ohlcv if len(x) >= 5]
        return closes
    except Exception:
        return []


def scan() -> Tuple[pd.DataFrame | None, List[Dict[str, Any]]]:
    ex = _build_exchange()
    ex.load_markets()

    # 1) Pull tickers in one batch
    tickers = ex.fetch_tickers()

    # 2) Filter to USD-ish quotes & sane bases
    rows: List[Tuple[str, float, float, float]] = []  # symbol, last, pct, vol_usd

    for sym, tkr in tickers.items():
        if not _want_symbol(sym):
            continue
        try:
            base, quote = sym.split("/")
            pct = _pct_from_ticker(tkr)
            if math.isnan(pct):
                continue
            last = float(tkr.get("last") or tkr.get("close") or 0.0)
            if last <= 0:
                continue
            vol_usd = _usd_volume_from_ticker(tkr, quote.upper())
            rows.append((sym, last, pct, vol_usd))
        except Exception:
            # keep scanning even if a symbol is messy
            continue

    if not rows:
        print("SUMMARY: MomentumSpike — 0 symbols passed basic parsing.")
        return None, []

    # 3) Primary filters
    filtered = [
        (s, last, pct, v)
        for (s, last, pct, v) in rows
        if pct >= MIN_24H_PCT and v >= MIN_BASE_VOL_USD
    ]

    # Sort by pct desc first (for shortlist)
    filtered.sort(key=lambda r: (r[2], r[3]), reverse=True)

    # 4) EMA check on the top-K short list to avoid pure wicks
    shortlist = filtered[: max(TOP_K_PRECHECK, MAX_RESULTS)]
    enriched: List[Dict[str, Any]] = []

    for s, last, pct, v in shortlist:
        closes = _fetch_ohlcv_close(ex, s, TIMEFRAME, EMA_MIN_BARS)
        if len(closes) < EMA_WINDOW + 2:
            # skip EMA gate if we can't fetch enough bars; keep but mark as unknown
            ema_val = float("nan")
            last_close = last
            above = None
        else:
            last_close = float(closes[-1])
            ema_val = _ema(closes, EMA_WINDOW)
            above = last_close > ema_val and (closes[-1] - closes[-2] >= 0)

        enriched.append(
            {
                "symbol": s,
                "last": round(float(last), 8),
                "pct_24h": round(float(pct), 4),
                "vol_usd_24h": round(float(v), 2),
                "ema_window": int(EMA_WINDOW),
                "last_close": round(float(last_close), 8),
                "ema": (round(float(ema_val), 8) if not math.isnan(ema_val) else None),
                "above_ema": (bool(above) if above is not None else None),
                "ts": int(time.time()),
            }
        )

        # small pause to be gentle with rate limits
        time.sleep(0.15)

    # 5) Final pick: prefer above_ema True, then pct desc, then volume
    def _rank_key(x: Dict[str, Any]):
        return (
            1 if x.get("above_ema") is True else 0,
            x.get("pct_24h", 0.0),
            x.get("vol_usd_24h", 0.0),
        )

    enriched.sort(key=_rank_key, reverse=True)
    final = enriched[:MAX_RESULTS]

    # 6) Save & summarize
    state_dir = _ensure_state_dir()
    out_csv = state_dir / "spike_candidates.csv"

    if pd is not None:
        df = pd.DataFrame(final)
        df.to_csv(out_csv, index=False)
    else:
        # minimal CSV writer if pandas isn't available
        import csv

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "symbol",
                    "last",
                    "pct_24h",
                    "vol_usd_24h",
                    "ema_window",
                    "last_close",
                    "ema",
                    "above_ema",
                    "ts",
                ],
            )
            w.writeheader()
            for row in final:
                w.writerow(row)
        df = None  # type: ignore

    # Pretty console summary
    print(
        f"SUMMARY: MomentumSpike — found {len(final)} candidates "
        f"(min %: {MIN_24H_PCT}, min vol: ${int(MIN_BASE_VOL_USD):,}, "
        f"EMA:{EMA_WINDOW}@{TIMEFRAME})."
    )
    for i, row in enumerate(final, 1):
        flag = "↑" if row.get("above_ema") else "·"
        print(
            f"  {i:>2}. {row['symbol']:>10}  "
            f"{flag}  {row['pct_24h']:>7.2f}%  "
            f"vol ${int(row['vol_usd_24h']):>10,}  "
            f"last {row['last']}"
        )

    print(f"ARTIFACT: wrote {out_csv}")

    # Also drop a tiny JSON summary for other steps if needed
    summary_json = {
        "count": len(final),
        "min_24h_pct": MIN_24H_PCT,
        "min_base_vol_usd": MIN_BASE_VOL_USD,
        "ema_window": EMA_WINDOW,
        "timeframe": TIMEFRAME,
        "candidates": final,
    }
    with open(state_dir / "spike_candidates.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    return df, final


def main():
    try:
        scan()
    except Exception as e:
        # Never crash the workflow — log and continue.
        print("ERROR: momentum_spike scan failed:", repr(e))
        # Still make empty artifacts so downstream steps don't break
        _ensure_state_dir()
        empty = pathlib.Path(".state") / "spike_candidates.csv"
        if not empty.exists():
            with open(empty, "w", encoding="utf-8") as f:
                f.write("symbol,last,pct_24h,vol_usd_24h,ema_window,last_close,ema,above_ema,ts\n")
        with open(pathlib.Path(".state") / "spike_candidates.json", "w", encoding="utf-8") as f:
            json.dump({"count": 0, "candidates": []}, f, indent=2)
        # Non-zero exit would fail the job — keep it zero.
        return


if __name__ == "__main__":
    main()
