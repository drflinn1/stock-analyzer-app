# trader/momentum_spike.py
import ccxt, pandas as pd, time

EXCH = ccxt.kraken()
SPIKE_PCT = 15       # minimum 24h gain %
MIN_USD_VOL = 15000  # skip illiquid
MAX_BUYS = 2
BUY_SIZE = 5.0       # USD per test buy

def top_spikes():
    tickers = EXCH.fetch_tickers()
    df = []
    for sym, t in tickers.items():
        if not sym.endswith("/USD"): 
            continue
        info = t.get('info', {})
        vol_usd = float(info.get('v', [0])[-1]) * t['last']
        change = (t['last'] - t['open']) / t['open'] * 100 if t['open'] else 0
        if change >= SPIKE_PCT and vol_usd >= MIN_USD_VOL:
            df.append((sym.replace("/USD", ""), change, vol_usd))
    top = sorted(df, key=lambda x: x[1], reverse=True)[:MAX_BUYS]
    return pd.DataFrame(top, columns=["symbol", "gain_pct", "vol_usd"])

def act_on_spikes():
    picks = top_spikes()
    if picks.empty:
        print("⚪ No spikes found.")
        return
    print("\n🚀 Spike candidates:")
    print(picks)
    for sym in picks["symbol"]:
        try:
            ticker = EXCH.fetch_ticker(f"{sym}/USD")
            price = ticker["last"]
            qty = round(BUY_SIZE / price, 5)
            print(f"Placing test buy: {sym}  qty={qty}  @${price:.4f}")
            # comment out real trade while dry-running
            # order = EXCH.create_market_buy_order(f"{sym}/USD", qty)
        except Exception as e:
            print(f"❌ {sym} error: {e}")

if __name__ == "__main__":
    act_on_spikes()
