ef momentum_scan(api):
    import time
    import numpy as np

    print("🔍 Step A: Early-Spike Momentum Scan")

    # Fetch tradeable USD pairs
    tradable = api.list_tradable_pairs()
    usd_pairs = [p for p in tradable if p.endswith("USD")]

    candidates = []

    for pair in usd_pairs:
        try:
            ohlc = api.get_ohlc(pair, interval=1)  # 1-minute candles
            closes = [float(c[4]) for c in ohlc]
            vols = [float(c[6]) for c in ohlc]

            if len(closes) < 30:
                continue

            # Fresh momentum – last 5 minutes only
            p_now  = closes[-1]
            p_1m   = closes[-2]
            p_3m   = closes[-4]
            p_5m   = closes[-6]

            fresh_momentum = (
                p_now > p_1m > p_3m > p_5m
            )

            if not fresh_momentum:
                continue  # skip — falling or weak trend

            # Volume check (15m baseline)
            avg_vol = np.mean(vols[-15:])
            vol_now = vols[-1]
            if vol_now < avg_vol * 1.25:
                continue  # skip — no volume confirmation

            # 1-hour percentage move (bigger context)
            p_60m = closes[-60] if len(closes) >= 60 else closes[0]
            pct_60m = (p_now - p_60m) / p_60m * 100

            candidates.append({
                "pair": pair,
                "momentum_5m": (p_now - p_5m) / p_5m * 100,
                "momentum_60m": pct_60m,
                "vol_boost": vol_now / avg_vol
            })

        except Exception as e:
            print(f"Scanner error {pair}: {e}")
            continue

    if not candidates:
        print("❌ No valid momentum candidates.")
        return None

    # Sort by strongest fresh 5m momentum first
    best = sorted(candidates, key=lambda x: x["momentum_5m"], reverse=True)[0]
    print(f"📈 Selected: {best['pair']} (fresh +{best['momentum_5m']:.2f}% in 5m)")

    return best["pair"]
