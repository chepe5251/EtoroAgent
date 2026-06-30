# Risk Discipline

## The Core Rule: Never Trade on a Single Signal
A single indicator or sentiment source is never enough to generate a trade.
Your thesis must cite at least 2 independent signals that point in the same direction.

### What counts as independent signals:
- RSI AND MACD (both technical, but independent calculations) ✅
- Indicators AND sentiment from any source ✅
- Two separate sentiment sources (Finnhub AND Reddit) ✅

### What does NOT count as 2 independent signals:
- RSI AND Bollinger Bands when both just say "price is low" (they're correlated in choppy markets) ⚠️
- Two Reddit posts saying the same thing ❌
- Same indicator over two different timeframes when you only have one timeframe's data ❌

## When "HOLD" is the Right Call

**Always prefer HOLD over a low-confidence trade.** Staying in cash is a position.

Return `action: "hold"` with low confidence when:
1. Technical signals contradict sentiment signals
2. Only one signal is present and it's not extreme (RSI between 35–65 + neutral sentiment)
3. Price is in a range (between EMA20 and EMA50, Bollinger Bands mid-range)
4. High uncertainty in the data (very few Reddit posts, no news, insufficient candles)
5. You've already examined the data and nothing clearly stands out

**Never manufacture a thesis** to have something to say. Honesty about uncertainty is valuable.

## Examples of Valid vs Invalid Theses

### ✅ VALID BUY thesis:
"BTC RSI(14) is at 28 (oversold). MACD histogram just turned positive (crossover 2 candles ago).
CryptoPanic shows bullish_count=12 vs bearish_count=3. Three independent signals aligned.
Confidence: 0.78. Stop-loss: 1.5× ATR = $840."
→ signals_used: ["indicators_full_analysis", "cryptopanic_get_sentiment_summary"]

### ✅ VALID HOLD thesis:
"TSLA RSI is 52 (neutral). MACD is slightly negative. Finnhub shows 48% bullish — ambiguous.
Reddit only has 3 posts (insufficient). No clear edge in either direction."
→ action: "hold", confidence: 0.35

### ❌ INVALID thesis (will be rejected by risk gate):
"BTC looks like it's going up based on market sentiment."
→ Only 1 vague signal. No citations. Reasoning too short. Will be rejected.

### ❌ INVALID thesis (will be rejected):
"The RSI is oversold at 25 so it should bounce."
→ Only 1 signal. No sentiment check done. Confidence must be < 0.65 here.
signals_used has only 1 entry — risk gate requires minimum 2.

## Stop-Loss Discipline
- Default stop: 1.5 × ATR from entry
- In high-volatility periods (ATR > 2% of price), use 2.0 × ATR or skip
- Never set stop below 0.5% of entry price (too tight, gets hunted)
- Never set stop beyond 5% of entry price (too wide, excessive risk per trade)
- If ATR is unavailable, default to 2% stop-loss

## Confidence Calibration Guide
| Situation | Max Confidence |
|---|---|
| 4+ signals aligned, low volatility | 0.92 |
| 3 signals aligned, moderate volatility | 0.82 |
| 2 signals aligned, normal conditions | 0.72 |
| 2 signals aligned, 1 contradicting | 0.62 (below threshold → HOLD) |
| 1 signal only, any strength | 0.55 (always below threshold → HOLD) |
| Contradictory signals | 0.30–0.45 → HOLD |
