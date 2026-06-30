# Technical Analysis Criteria

## RSI (14 periods)
- **< 30**: Oversold. Potential reversal signal upward. Stronger if combined with price at support or positive sentiment.
- **> 70**: Overbought. Potential reversal signal downward. Stronger if price near resistance or negative sentiment.
- **30–70**: Neutral zone. RSI alone is not a trade signal in this range.
- **Divergence**: If price makes new highs but RSI doesn't, this is bearish divergence (and vice versa). Weight this heavily.

## MACD (12, 26, 9)
- **MACD line crosses above signal line (histogram turns positive)**: Bullish momentum confirmation.
- **MACD line crosses below signal line (histogram turns negative)**: Bearish momentum confirmation.
- **Histogram growing in the same direction as the cross**: Strong momentum. Shrinking histogram = momentum fading.
- **MACD alone is NOT sufficient** — it's a confirmation tool, not a primary signal.

## EMA 20 / EMA 50 crossover
- **EMA20 > EMA50** (golden cross): Bullish trend. A BUY signal becomes stronger here.
- **EMA20 < EMA50** (death cross): Bearish trend. A SELL signal becomes stronger here.
- **Price above EMA20 and EMA50**: Strong uptrend. Price below both: Strong downtrend.
- **Price between EMA20 and EMA50**: Consolidation — be cautious about trend signals.

## Bollinger Bands (20 periods, 2σ)
- **Price touches lower band**: Oversold relative to recent volatility. Possible bounce. Only valid if RSI also oversold.
- **Price touches upper band**: Overbought relative to recent volatility. Possible pullback.
- **Band squeeze (bandwidth < 0.02)**: Low volatility precedes high volatility. Direction unknown — wait for breakout.
- **Band expansion**: Trend is accelerating. Trade in the direction of the move.

## ATR (14 periods) — for stop-loss sizing
- **Stop-loss rule**: Set initial stop at **1.5 × ATR** from entry price.
- **If ATR is elevated** (> 2% of price): Consider reducing position size or skipping the trade — volatility risk is high.
- **Default ATR multiple**: 1.5. In strongly trending markets, use 2.0 to give the trade more room.

## Relative Volume
- **> 1.5**: Volume is 50% above average — confirms the move. Weight the signal more.
- **< 0.7**: Low volume — the move may not be sustained. Reduce confidence.
- **Volume spike without clear price direction**: Often signals indecision or manipulation. Be cautious.

## Combining Signals (Required)
Always require at least 2 technical signals pointing in the same direction before including them in `signals_used`:
- RSI oversold + MACD histogram turning positive = valid bullish combination
- EMA death cross + RSI neutral = weak signal, not enough alone
- RSI overbought + price at Bollinger upper + high volume = strong bearish combination
