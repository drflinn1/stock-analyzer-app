# engine.py — core logic (no Streamlit UI)
# Keep this file import-only; login & UI happen in app.py

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

# Robinhood (used only when live trading is enabled & authenticated in app.py)
try:
    import robin_stocks.robinhood as rh
    from robin_stocks.robinhood import orders as rh_orders
    from robin_stocks.robinhood import crypto as rh_crypto
    from robin_stocks.robinhood import account as rh_account
    from robin_stocks.robinhood import stocks as rh_stocks
except Exception:
    rh = None
    rh_orders = None
    rh_crypto = None
    rh_account = None
    rh_stocks = None

# ---------- Universes ----------

def load_sp500_symbols() -> List[str]:
    """S&P500 with fallbacks; no Streamlit caching here to keep engine pure."""
    # 1) Wikipedia (best effort)
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        if isinstance(tables, list) and len(tables):
            df = tables[0]
            syms = (
                df["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False).tolist()
            )
            syms = [s for s in dict.fromkeys(syms) if s.isascii() and 1 <= len(s) <= 6]
            if syms:
                return syms
    except Exception:
        pass
    # 2) CSV mirrors
    for url in [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
        "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/constituents.csv",
    ]:
        try:
            df = pd.read_csv(url)
            col = next((c for c in ["Symbol", "Ticker", "symbol", "ticker"] if c in df.columns), None)
            if col:
                syms = df[col].astype(str).str.upper().str.replace(".", "-", regex=False).tolist()
                syms = [s for s in dict.fromkeys(syms) if s.isascii() and 1 <= len(s) <= 6]
                if syms:
                    return syms
        except Exception:
            continue
    # 3) baked fallback
    return ["AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "BRK-B", "TSLA", "JPM"]

def clean_manual_list(raw: str) -> List[str]:
    return [t.strip().upper() for t in (raw or "").split(",") if t.strip()]

DEFAULT_CRYPTO = ["BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "USDT-USD"]

def load_rh_crypto_pairs() -> List[str]:
    """Return symbols in Yahoo-style (e.g., ETH-USD). Fallback to DEFAULT_CRYPTO."""
    try:
        pairs = []
        if rh_crypto is not None and hasattr(rh_crypto, "get_crypto_currency_pairs"):
            data = rh_crypto.get_crypto_currency_pairs()
            if isinstance(data, list):
                for p in data:
                    base = (p.get("asset_currency", {}) or {}).get("code")
                    if base:
                        pairs.append(f"{base}-USD")
        if not pairs:
            pairs = DEFAULT_CRYPTO.copy()
        return sorted(list(dict.fromkeys(pairs)))
    except Exception:
        return DEFAULT_CRYPTO.copy()

def build_universe(eq_src: str, manual_raw: str, include_c: bool) -> List[str]:
    eq = load_sp500_symbols() if eq_src == "S&P 500 (auto)" else clean_manual_list(manual_raw)
    cr = load_rh_crypto_pairs() if include_c else []
    return list(dict.fromkeys(eq + cr))

# ---------- Data & Scoring ----------

def fetch_returns(symbols: List[str], lookbacks: List[int]) -> pd.DataFrame:
    if not symbols:
        cols = ["Ticker"] + [f"R{lb}" for lb in lookbacks]
        return pd.DataFrame(columns=cols)

    def _calc(sym: str, df1: pd.DataFrame) -> Dict[str, float]:
        out = {"Ticker": sym}
        if df1 is None or df1.empty or "Close" not in df1:
            for lb in lookbacks: out[f"R{lb}"] = np.nan
            return out
        close = df1["Close"].dropna()
        for lb in lookbacks:
            try:
                if len(close) >= (lb + 1):
                    r = (close.iloc[-1] / close.iloc[-(lb + 1)] - 1.0) * 100.0
                else:
                    r = np.nan
            except Exception:
                r = np.nan
            out[f"R{lb}"] = float(r) if pd.notna(r) else np.nan
        return out

    rows: List[Dict] = []
    try:
        df = yf.download(symbols, period="60d", interval="1d", progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            for sym in (df["Close"].columns if "Close" in df else symbols):
                dfx = df.xs(sym, axis=1, level=1)[["Close"]] if "Close" in df else None
                rows.append(_calc(str(sym), dfx))
        else:
            rows.append(_calc(str(symbols[0]), df))
    except Exception:
        for s in symbols:
            try:
                dfx = yf.download(s, period="60d", interval="1d", progress=False, threads=False)
            except Exception:
                dfx = None
            rows.append(_calc(s, dfx))
    return pd.DataFrame(rows)

def score_momentum(df: pd.DataFrame, weights: Dict[int, float] | None = None) -> pd.DataFrame:
    if weights is None:
        weights = {1: 0.6, 5: 0.3, 20: 0.1}
    sc = df.copy()
    for lb, w in weights.items():
        col = f"R{lb}"
        if col not in sc: sc[col] = np.nan
        mu, sd = sc[col].mean(skipna=True), sc[col].std(skipna=True)
        sc[f"Z{lb}"] = 0.0 if pd.isna(sd) or sd == 0 else (sc[col] - mu) / sd
    sc["Score"] = sum(sc.get(f"Z{lb}", 0.0) * w for lb, w in weights.items())
    return sc

# ---------- Symbols & Quotes ----------

def yahoo_to_rh_stock(sym: str) -> str:
    s = str(sym or "").upper().strip()
    return s if s.endswith("-USD") else s.replace("-", ".")

def rh_to_yahoo_stock(sym: str) -> str:
    s = str(sym or "").upper().strip()
    return s.replace(".", "-") if "." in s else s

def symbol_to_rh_crypto(sym: str) -> str:
    return sym.split("-")[0] if "-" in sym else sym

def normalize_dollars(dollars: float, min_amt: float) -> float:
    d = max(float(dollars), float(min_amt))
    cents = int(np.ceil(d * 100.0 - 1e-9))
    return cents / 100.0

def _get_stock_last_price(sym: str) -> float:
    try:
        if rh_stocks is not None and hasattr(rh_stocks, "get_latest_price"):
            px = rh_stocks.get_latest_price(yahoo_to_rh_stock(sym), includeExtendedHours=True)
            if isinstance(px, list) and px:
                return float(px[0])
    except Exception:
        pass
    try:
        fi = getattr(yf.Ticker(sym), "fast_info", None) or {}
        p = fi.get("last_price") or fi.get("last_close")
        if p: return float(p)
    except Exception:
        pass
    return 0.0

def _get_crypto_last_price(sym_no_usd: str) -> float:
    try:
        if rh_crypto is not None and hasattr(rh_crypto, "get_crypto_quote"):
            q = rh_crypto.get_crypto_quote(sym_no_usd)
            ask = float(q.get("ask_price", 0) or 0); bid = float(q.get("bid_price", 0) or 0)
            if ask and bid: return (ask + bid) / 2.0
            return float(q.get("mark_price", 0) or ask or bid or 0)
    except Exception:
        pass
    try:
        fi = getattr(yf.Ticker(f"{sym_no_usd}-USD"), "fast_info", None) or {}
        p = fi.get("last_price") or fi.get("last_close")
        if p: return float(p)
    except Exception:
        pass
    return 0.0

# ---------- Orders ----------

def place_stock_order(symbol: str, side: str, dollars: float, live: bool, min_order: float,
                      use_limit: bool = False, limit_bps: int = 25) -> Tuple[str, float, str]:
    base = normalize_dollars(dollars, min_order)
    if base < 0.01: return ("skipped (too small)", 0.0, "")
    if not live:     return ("simulated", 0.0, "")
    rh_symbol = yahoo_to_rh_stock(symbol)

    # experimental limit path
    if use_limit and rh_orders is not None:
        try:
            px = _get_stock_last_price(symbol)
            if px and px > 0:
                qty = max(base / px, 0.0001); qty = float(np.round(qty, 6))
                buff = float(limit_bps) / 10000.0
                if side.upper() == "BUY":
                    limit_price = float(np.round(px * (1 + buff), 4))
                    fn = getattr(rh_orders, "order_buy_fractional_limit", None) or getattr(rh_orders, "order_buy_limit")
                    res = fn(rh_symbol, qty, limit_price)
                else:
                    limit_price = float(np.round(px * (1 - buff), 4))
                    fn = getattr(rh_orders, "order_sell_fractional_limit", None) or getattr(rh_orders, "order_sell_limit")
                    res = fn(rh_symbol, qty, limit_price)
                oid = (res.get("id") or res.get("order_id")) if isinstance(res, dict) else ""
                return ("placed", 0.0, oid)
        except Exception:
            pass

    for attempt in range(3):
        amt = normalize_dollars(base * (1.02 ** attempt), min_order)
        try:
            if side.upper() == "BUY":
                res = rh_orders.order_buy_fractional_by_price(rh_symbol, amt)
            else:
                res = rh_orders.order_sell_fractional_by_price(rh_symbol, amt)
            oid = (res.get("id") or res.get("order_id")) if isinstance(res, dict) else ""
            return ("placed", 0.0, oid)
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ["too small", "minimum", "tick", "notional"]):
                continue
            return (f"error: {e}", 0.0, "")
    return ("error: minimum notional too small after retries", 0.0, "")

def place_crypto_order(symbol: str, side: str, dollars: float, live: bool, min_order: float,
                       use_limit: bool = True, limit_bps: int = 20) -> Tuple[str, float, str]:
    base = normalize_dollars(dollars, min_order)
    if base < 0.01: return ("skipped (too small)", 0.0, "")
    if not live:     return ("simulated", 0.0, "")
    sym = symbol_to_rh_crypto(symbol)

    if use_limit and rh_orders is not None:
        try:
            px = _get_crypto_last_price(sym)
            if px and px > 0:
                qty = max(base / px, 0.00000001); qty = float(np.round(qty, 8))
                buff = float(limit_bps) / 10000.0
                if side.upper() == "BUY":
                    limit_price = float(np.round(px * (1 + buff), 2))
                    res = rh_orders.order_buy_crypto_limit(sym, qty, limit_price)
                else:
                    limit_price = float(np.round(px * (1 - buff), 2))
                    res = rh_orders.order_sell_crypto_limit(sym, qty, limit_price)
                oid = (res.get("id") or res.get("order_id")) if isinstance(res, dict) else ""
                return ("placed", 0.0, oid)
        except Exception:
            pass

    buy_candidates  = ["order_buy_crypto_by_price", "order_buy_crypto_by_dollar", "order_buy_crypto_by_dollars", "buy_crypto_by_price"]
    sell_candidates = ["order_sell_crypto_by_price", "order_sell_crypto_by_dollar", "order_sell_crypto_by_dollars", "sell_crypto_by_price"]
    for attempt in range(3):
        amt = normalize_dollars(base * (1.02 ** attempt), min_order)
        try:
            fn = None
            for name in (buy_candidates if side.upper() == "BUY" else sell_candidates):
                fn = getattr(rh_crypto, name, None) or getattr(rh, name, None)
                if callable(fn): break
            res = fn(sym, amt) if callable(fn) else {}
            oid = (res.get("id") or res.get("order_id")) if isinstance(res, dict) else ""
            return ("placed", 0.0, oid)
        except Exception as e:
            if any(k in str(e).lower() for k in ["too small", "minimum", "tick", "notional"]):
                continue
            return (f"error: {e}", 0.0, "")
    return ("error: minimum notional too small after retries", 0.0, "")

# ---------- Holdings / BP ----------

def get_holdings() -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    if rh_account is None: return out
    # stocks
    try:
        if hasattr(rh_account, "build_holdings"):
            h = rh_account.build_holdings()
            if isinstance(h, dict):
                for sym, info in h.items():
                    value = float(info.get("equity", 0) or 0)
                    qty   = float(info.get("quantity", 0) or 0)
                    out[rh_to_yahoo_stock(str(sym))] = {"type": "stock", "value": value, "qty": qty}
    except Exception:
        pass
    # crypto
    try:
        if rh_crypto is not None and hasattr(rh_crypto, "get_crypto_positions"):
            pos = rh_crypto.get_crypto_positions()
            if isinstance(pos, list):
                for p in pos:
                    qty = float(p.get("quantity", 0) or 0)
                    sym = (p.get("currency", {}) or {}).get("code") or ""
                    if qty > 0 and sym:
                        price = 0.0
                        try:
                            q = rh_crypto.get_crypto_quote(sym)
                            price = float(q.get("mark_price", 0) or q.get("ask_price", 0) or 0)
                        except Exception:
                            pass
                        out[f"{sym}-USD".upper()] = {"type": "crypto", "value": qty * price, "qty": qty}
    except Exception:
        pass
    return out

def get_buying_power() -> float:
    bp = 0.0
    try:
        if rh_account is not None:
            prof = None
            for nm in ["load_account_profile", "build_user_profile", "load_phoenix_account"]:
                fn = getattr(rh_account, nm, None)
                if callable(fn):
                    try:
                        prof = fn(); 
                        if prof: break
                    except Exception:
                        continue
            if isinstance(prof, dict):
                keys = ["buying_power", "cash", "portfolio_cash", "cash_available_for_withdrawal"]
                for key in keys:
                    try:
                        val = prof.get(key); 
                        if val is not None and float(val) > 0: bp = max(bp, float(val))
                    except Exception:
                        pass
                for v in prof.values():
                    if isinstance(v, dict):
                        for key in keys:
                            try:
                                if key in v and float(v[key]) > 0: bp = max(bp, float(v[key]))
                            except Exception:
                                pass
    except Exception:
        pass
    try:
        if rh_crypto is not None and hasattr(rh_crypto, "get_crypto_account_info"):
            cp = rh_crypto.get_crypto_account_info()
            if isinstance(cp, dict):
                for key in ["cash", "buying_power", "available_cash"]:
                    try:
                        val = cp.get(key)
                        if val is not None and float(val) > 0: bp = max(bp, float(val))
                    except Exception:
                        pass
    except Exception:
        pass
    return float(max(bp, 0.0))

# ---------- Open orders & cancel ----------

def _crypto_pair_maps():
    id2sym, sym2id = {}, {}
    try:
        if rh_crypto is not None and hasattr(rh_crypto, "get_crypto_currency_pairs"):
            pairs = rh_crypto.get_crypto_currency_pairs()
            if isinstance(pairs, list):
                for p in pairs:
                    pid = p.get("id") or p.get("uuid") or p.get("currency_pair_id")
                    base = (p.get("asset_currency") or {}).get("code") or p.get("symbol") or ""
                    quote = (p.get("quote_currency") or {}).get("code") or "USD"
                    if base:
                        sym = f"{base}-{quote}"
                        if pid: id2sym[pid] = sym
                        sym2id[sym] = pid or ""
    except Exception:
        pass
    return id2sym, sym2id

def _stock_symbol_from_instrument(url: str) -> str:
    try:
        if rh_stocks is not None:
            for nm in ["get_instrument_by_url", "get_instruments_by_url"]:
                fn = getattr(rh_stocks, nm, None)
                if callable(fn):
                    inst = fn(url)
                    if isinstance(inst, dict):
                        sym = inst.get("symbol") or ""
                        if sym: return rh_to_yahoo_stock(sym)
    except Exception:
        pass
    try:
        fn = getattr(rh, "get_symbol_by_url", None)
        if callable(fn):
            sym = fn(url)
            if sym: return rh_to_yahoo_stock(sym)
    except Exception:
        pass
    return ""

def get_open_orders_raw(live_trading: bool, login_ok: bool):
    open_list = []
    if not (live_trading and login_ok): return open_list
    try:
        if rh_orders is not None and hasattr(rh_orders, "get_all_open_stock_orders"):
            s_open = rh_orders.get_all_open_stock_orders()
            if isinstance(s_open, list):
                open_list.extend([{"type": "stock", **(o if isinstance(o, dict) else {"raw": str(o)})} for o in s_open])
    except Exception:
        pass
    try:
        if rh_crypto is not None and hasattr(rh_crypto, "get_crypto_open_orders"):
            c_open = rh_crypto.get_crypto_open_orders()
            if isinstance(c_open, list):
                open_list.extend([{"type": "crypto", **(o if isinstance(o, dict) else {"raw": str(o)})} for o in c_open])
    except Exception:
        pass
    return open_list

def open_orders_table(live_trading: bool, login_ok: bool) -> List[Dict]:
    rows = []
    for o in get_open_orders_raw(live_trading, login_ok):
        if not isinstance(o, dict): continue
        typ = o.get("type", "")
        side = o.get("side") or o.get("direction") or ""
        sym = o.get("symbol") or o.get("chain_symbol") or ""
        if not sym:
            inst = o.get("instrument")
            if inst: sym = _stock_symbol_from_instrument(inst)
        if not sym:
            id2sym, _ = _crypto_pair_maps()
            cp = o.get("currency_pair_id") or o.get("currency_pair") or o.get("pair")
            if cp: sym = id2sym.get(cp, str(cp))
        if isinstance(sym, str) and sym.endswith("USD") and "-" not in sym:
            sym = sym.replace("USD", "-USD")
        state   = o.get("state") or o.get("status") or ""
        created = o.get("created_at") or o.get("last_transaction_at") or o.get("submitted_at") or ""
        oid     = o.get("id") or o.get("order_id") or ""
        notional= o.get("notional") or o.get("price") or o.get("average_price") or o.get("quantity") or ""
        rows.append({"Type": typ, "Side": side, "Symbol": sym, "Notional/Qty": notional, "State": state, "Created": created, "OrderID": oid})
    return rows

def _parse_when(s: str):
    try:
        return pd.to_datetime(s, utc=True, errors="coerce")
    except Exception:
        return pd.NaT

def cancel_open_orders(*, live_trading: bool, login_ok: bool, older_than_minutes: int | None = None, cancel_all: bool = False) -> int:
    if not (live_trading and login_ok): return 0
    now = pd.Timestamp.utcnow().tz_localize("UTC")
    cnt = 0
    for o in get_open_orders_raw(live_trading, login_ok):
        if not isinstance(o, dict): continue
        oid = o.get("id") or o.get("order_id")
        if not oid:        continue
        if not cancel_all and older_than_minutes is not None:
            when = _parse_when(o.get("created_at") or o.get("last_transaction_at") or o.get("submitted_at") or "")
            if pd.isna(when): continue
            if (now - when).total_seconds() / 60.0 < float(older_than_minutes):
                continue
        typ = o.get("type") or ("crypto" if (o.get("currency_pair") or o.get("currency_pair_id")) else "stock")
        try:
            if typ == "crypto":
                (getattr(rh_orders, "cancel_crypto_order", None) or getattr(rh_crypto, "cancel_order", None) or getattr(rh, "cancel_order", None))(oid)
            else:
                (getattr(rh_orders, "cancel_stock_order", None) or getattr(rh, "cancel_order", None))(oid)
            cnt += 1
        except Exception:
            continue
    return cnt

# ---------- One run ----------

def run_once(*,
             live_trading: bool,
             login_ok: bool,
             universe_src: str,
             raw_tickers: str,
             include_crypto: bool,
             alloc_mode: str,
             fixed_per_trade: float,
             prop_total_budget: float,
             min_per_order: float,
             n_picks: int,
             use_crypto_limits: bool,
             crypto_limit_bps: int,
             use_stock_limits: bool,
             stock_limit_bps: int,
             auto_bp: bool,
             bp_pct: int,
             max_buy_orders: int,
             max_buy_notional: float,
             ) -> Dict:
    # Universe & scores
    universe = build_universe(universe_src, raw_tickers, include_crypto)
    lookbacks = [1, 5, 20]
    ret_df = fetch_returns(universe, lookbacks)
    scored = score_momentum(ret_df, weights={1: 0.6, 5: 0.3, 20: 0.1})
    picks = scored.dropna(subset=["Score"]).sort_values("Score", ascending=False).head(int(n_picks)).reset_index(drop=True)
    if picks.empty:
        return {"picks": picks, "sell_rows": [], "buy_rows": [], "budget_used": 0.0, "bp_seen": 0.0, "auto_budget": False}

    # SELL rule (out of top-N OR score < 0)
    current = get_holdings() if (live_trading and login_ok) else {}
    in_top = set(picks["Ticker"].astype(str).str.upper().tolist())
    score_map = {str(t).upper(): (float(s) if pd.notna(s) else 0.0) for t, s in zip(scored["Ticker"].astype(str), scored["Score"])}
    sell_rows: List[Dict] = []

    for sym, info in current.items():
        t_upper = str(sym).upper()
        sym_score = float(score_map.get(t_upper, 0.0))
        out_of_top = t_upper not in in_top
        neg_score  = sym_score < 0
        if out_of_top or neg_score:
            dollars = float(info.get("value", 0.0))
            if dollars >= min_per_order:
                if info.get("type") == "crypto":
                    status, executed, oid = place_crypto_order(symbol_to_rh_crypto(t_upper), "SELL", dollars, live_trading and login_ok, min_per_order,
                                                               use_limit=use_crypto_limits, limit_bps=crypto_limit_bps)
                else:
                    status, executed, oid = place_stock_order(t_upper, "SELL", dollars, live_trading and login_ok, min_per_order,
                                                              use_limit=use_stock_limits, limit_bps=stock_limit_bps)
                sell_rows.append({
                    "Ticker": t_upper, "Action": "SELL",
                    "Reason": "negative_score" if neg_score and not out_of_top else ("out_of_topN" if out_of_top else "negative_score"),
                    "Alloc$": round(dollars, 2), "OrderID": oid, "Status": status,
                    "Time": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                })

    # BUY allocs
    auto_budget = False
    bp_seen = 0.0
    if alloc_mode == "Fixed $ per trade":
        per_trade = float(max(min_per_order, fixed_per_trade))
        buy_allocs = {t: per_trade for t in in_top}
        budget_used = per_trade * len(in_top)
    else:
        budget = float(prop_total_budget)
        if auto_bp and live_trading and login_ok:
            bp_seen = get_buying_power()
            if bp_seen and bp_seen > 0:
                budget = max(min_per_order, round((bp_seen * float(bp_pct) / 100.0), 2))
                auto_budget = True
        w = np.clip(picks["R1"].fillna(0).astype(float), 0, None)
        wsum = float(w.sum())
        buy_allocs = {}
        if wsum <= 0:
            each = max(min_per_order, budget / max(1, len(picks)))
            for t in in_top: buy_allocs[t] = each
        else:
            for t, wi in zip(picks["Ticker"].tolist(), w.tolist()):
                buy_allocs[str(t).upper()] = float(max(min_per_order, (wi / wsum) * float(budget)))
        budget_used = float(sum(buy_allocs.values()))

    # BUY loop with safety caps
    buy_rows: List[Dict] = []
    buy_count = 0; buy_notional = 0.0
    for _, r in picks.iterrows():
        t = str(r["Ticker"]).upper()
        score = float(r["Score"]) if pd.notna(r["Score"]) else 0.0
        if score < 0: continue
        dollars = float(buy_allocs.get(t, 0.0))
        if dollars < min_per_order: continue
        if buy_count >= int(max_buy_orders) or (buy_notional + dollars) > float(max_buy_notional):
            status, executed, oid = ("skipped (safety cap)", 0.0, "")
        else:
            is_crypto = t.endswith("-USD")
            if is_crypto:
                status, executed, oid = place_crypto_order(symbol_to_rh_crypto(t), "BUY", dollars, live_trading and login_ok, min_per_order,
                                                           use_limit=use_crypto_limits, limit_bps=crypto_limit_bps)
            else:
                status, executed, oid = place_stock_order(t, "BUY", dollars, live_trading and login_ok, min_per_order,
                                                          use_limit=use_stock_limits, limit_bps=stock_limit_bps)
            if status.startswith("placed") or status == "simulated":
                buy_count += 1; buy_notional = float(np.round(buy_notional + dollars, 2))
        buy_rows.append({
            "Ticker": t, "Action": "BUY", "Alloc$": round(dollars, 2), "OrderID": oid, "Status": status,
            "R1%": round(float(r.get("R1", 0.0)), 3), "R5%": round(float(r.get("R5", 0.0)), 3),
            "R20%": round(float(r.get("R20", 0.0)), 3), "Score": round(score, 4),
            "Time": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        })

    return {
        "picks": picks,
        "sell_rows": sell_rows,
        "buy_rows":  buy_rows,
        "budget_used": float(np.round(budget_used, 2)),
        "bp_seen": float(np.round(bp_seen, 2)),
        "auto_budget": bool(auto_budget),
        "buy_count": int(buy_count),
        "buy_notional": float(np.round(buy_notional, 2)),
    }
