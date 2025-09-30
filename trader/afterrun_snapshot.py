# trader/afterrun_snapshot.py
# Prints a concise snapshot after a run.
# If API keys are missing, it skips private calls and exits cleanly.

from __future__ import annotations
import os, datetime as dt

try:
    import ccxt  # type: ignore
except Exception as e:
    print(f"\x1b[31m[ERROR]\x1b[0m ccxt is required: {e}")
    raise SystemExit(0)

GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
RESET = "\x1b[0m"

EXCHANGE_ID = os.getenv("EXCHANGE_ID", "kraken")

def get_exchange():
    api_key = os.getenv("CCXT_API_KEY") or ""
    api_secret = os.getenv("CCXT_API_SECRET") or ""
    api_password = os.getenv("CCXT_API_PASSWORD") or ""
    has_keys = bool(api_key and api_secret)
    cls = getattr(ccxt, EXCHANGE_ID)
    ex = cls({
        "apiKey": api_key if has_keys else None,
        "secret": api_secret if has_keys else None,
        "password": api_password if has_keys else None,
        "enableRateLimit": True,
    })
    ex.load_markets()
    return ex, has_keys

def fmt_ts(ms: int) -> str:
    return dt.datetime.utcfromtimestamp(ms/1000.0).strftime("%Y-%m-%d %H:%M:%S UTC")

def main():
    ex, has_keys = get_exchange()

    if not has_keys:
        print(f"{YELLOW}SNAPSHOT: Skipping balances/positions/trades — no API keys in env for {EXCHANGE_ID}.{RESET}")
        print(f"{YELLOW}Tip:{RESET} add CCXT_API_KEY / CCXT_API_SECRET (and CCXT_API_PASSWORD if needed) to repo secrets "
              "and this step will start showing holdings and trades automatically.")
        return

    # --- Balances ---
    try:
        bal = ex.fetch_balance()
        usd = 0.0
        for k in ("USD", "ZUSD"):
            usd += float(bal.get("total", {}).get(k, 0.0))
        print(f"{YELLOW}SNAPSHOT: USD balance ≈ {usd:.2f}{RESET}")
    except Exception as e:
        print(f"{RED}[WARN]{RESET} fetch_balance failed: {e}")
        bal = {"total": {}}

    # --- Positions ---
    totals = bal.get("total", {})
    positions = []
    for asset, qty in sorted(totals.items()):
        try:
            qty = float(qty or 0.0)
        except Exception:
            qty = 0.0
        if asset in ("USD", "ZUSD") or qty <= 1e-9:
            continue
        price = None
        for sym in (f"{asset}/USD", f"{asset}/USDT"):
            if sym in ex.markets:
                try:
                    t = ex.fetch_ticker(sym)
                    price = t.get("last") or t.get("close")
                    break
                except Exception:
                    pass
        positions.append((asset, qty, price))

    if positions:
        print(f"{YELLOW}OPEN POSITIONS:{RESET}")
        for asset, qty, px in positions:
            val = (qty * px) if px else None
            val_s = f" ≈ ${val:.2f}" if val is not None else ""
            print(f"  - {asset}: {qty:.8f}{val_s}")
    else:
        print(f"{YELLOW}OPEN POSITIONS: none or unavailable{RESET}")

    # --- Recent trades (last 10) across common symbols ---
    print(f"{YELLOW}RECENT TRADES (last 10):{RESET}")
    trades_all = []
    for base in ("BTC","ETH","SOL","DOGE"):
        for quote in ("USD","USDT"):
            sym = f"{base}/{quote}"
            if sym in ex.markets:
                try:
                    trades = ex.fetch_my_trades(sym, limit=10)
                    trades_all.extend(trades)
                except Exception:
                    pass

    trades_all.sort(key=lambda t: t.get("timestamp") or 0, reverse=True)
    trades_all = trades_all[:10]

    if not trades_all:
        print("  (no recent trades reported)")
        return

    for t in trades_all:
        side = (t.get("side") or "").upper()
        color = GREEN if side == "BUY" else RED if side == "SELL" else RESET
        sym = t.get("symbol","?")
        amount = float(t.get("amount",0.0) or 0.0)
        price = float(t.get("price",0.0) or 0.0)
        ts = fmt_ts(t.get("timestamp",0))
        fee = t.get("fee",{}) or {}
        fee_str = f" fee={fee.get('cost')} {fee.get('currency','')}".strip()
        print(f"  {color}{side:<4}{RESET} {sym:<10} qty={amount:.8f} @ {price:.6f}  {ts}  {fee_str}")

if __name__ == "__main__":
    main()
