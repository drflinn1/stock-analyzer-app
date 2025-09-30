# trader/afterrun_snapshot.py
# Prints a concise snapshot after a run:
# - Balances (USD + top coins)
# - Open positions (estimated)
# - Last 10 trades (BUY/SELL, symbol, qty, price, time)

from __future__ import annotations
import os, math, datetime as dt

try:
    import ccxt  # type: ignore
except Exception as e:
    print(f"\x1b[31m[ERROR]\x1b[0m ccxt is required: {e}")
    raise SystemExit(1)

GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
RESET = "\x1b[0m"

EXCHANGE_ID = os.getenv("EXCHANGE_ID", "kraken")
USD_KEYS = ("USD", "ZUSD")  # Kraken

WATCH = ["BTC", "ETH", "SOL", "DOGE"]

def get_exchange():
    cls = getattr(ccxt, EXCHANGE_ID)
    ex = cls({
        "apiKey": os.getenv("CCXT_API_KEY"),
        "secret": os.getenv("CCXT_API_SECRET"),
        "password": os.getenv("CCXT_API_PASSWORD"),
        "enableRateLimit": True,
    })
    ex.load_markets()
    return ex

def fmt_ts(ms: int) -> str:
    return dt.datetime.utcfromtimestamp(ms/1000.0).strftime("%Y-%m-%d %H:%M:%S UTC")

def main():
    ex = get_exchange()

    # Balances
    bal = ex.fetch_balance()
    usd = sum(bal["total"].get(k, 0.0) for k in USD_KEYS)
    print(f"{YELLOW}SNAPSHOT: USD balance ≈ {usd:.2f}{RESET}")

    # Quick positions guess = non-USD assets with non-tiny totals
    positions = []
    for asset, qty in bal["total"].items():
        if asset in USD_KEYS:
            continue
        if qty and qty > 1e-9:
            # try to value it vs USD if a market exists
            sym1 = f"{asset}/USD"
            sym2 = f"{asset}/USDT"
            price = None
            for s in (sym1, sym2):
                if s in ex.markets:
                    try:
                        ticker = ex.fetch_ticker(s)
                        price = ticker.get("last") or ticker.get("close")
                        break
                    except Exception:
                        pass
            positions.append((asset, qty, price))

    if positions:
        print(f"{YELLOW}OPEN POSITIONS:{RESET}")
        for asset, qty, px in positions:
            val = (qty * px) if (px and px == px) else None
            val_s = f" ≈ ${val:.2f}" if val is not None else ""
            print(f"  - {asset}: {qty:.8f}{val_s}")
    else:
        print(f"{YELLOW}OPEN POSITIONS: none{RESET}")

    # Recent trades (last 10) across a few symbols
    print(f"{YELLOW}RECENT TRADES (last 10):{RESET}")
    trades_all = []
    for base in ["BTC","ETH","SOL","DOGE"]:
        for quote in ["USD","USDT"]:
            sym = f"{base}/{quote}"
            if sym in ex.markets:
                try:
                    trades = ex.fetch_my_trades(sym, limit=10)
                    trades_all.extend(trades)
                except Exception:
                    pass
    # sort by time desc and take top 10
    trades_all.sort(key=lambda t: t.get("timestamp") or 0, reverse=True)
    trades_all = trades_all[:10]

    if not trades_all:
        print("  (no recent trades reported by exchange API)")
        return

    for t in trades_all:
        side = t.get("side","").upper()
        color = GREEN if side == "BUY" else RED if side == "SELL" else RESET
        sym = t.get("symbol","?")
        amount = t.get("amount",0.0)
        price = t.get("price",0.0)
        ts = fmt_ts(t.get("timestamp",0))
        fee = t.get("fee",{}) or {}
        fee_str = f" fee={fee.get('cost')} {fee.get('currency','')}".strip()
        print(f"  {color}{side:<4}{RESET} {sym:<10} qty={amount:.8f} @ {price:.6f}  {ts}  {fee_str}")
        
if __name__ == "__main__":
    main()
