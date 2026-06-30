# Swing Trading Guidelines (5–20 Day Horizon)

## What Swing Trading Is

Swing trading captures multi-day price movements driven by momentum shifts, trend resumptions, or mean reversion after extremes. It is NOT:
- Scalping (minutes/hours)
- Position trading (months/years)
- News gambling (random catalyst chasing)

A swing trade needs a clear **structural reason** to exist and a clear **exit condition** defined upfront.

## When to Enter

**Look for alignment between at least 2 of these:**

### 1. RSI Mean Reversion (D1 timeframe)
- RSI(14) < 30: potential bullish swing entry after exhaustion
- RSI(14) > 70: potential bearish swing entry after exhaustion
- RSI must be RECOVERING, not just at the extreme (wait for RSI turning, not just touching)
- Stronger signal if RSI diverges from price (price makes new low but RSI does not)

### 2. EMA Structure
- EMA20 crossing above EMA50 (D1): bullish momentum shift → BUY bias
- EMA20 crossing below EMA50 (D1): bearish momentum shift → SELL bias
- Price reclaiming EMA20 after a pullback in an uptrend: continuation entry
- Cross must be recent (last 3-5 candles); stale crosses are weak signals

### 3. Volume Confirmation
- Entry candle volume > 1.5× the 20-day average: institutional interest
- Low-volume breakouts often fail — do NOT enter on them alone
- High volume on RSI-oversold day = exhaustion sell (bullish reversal signal)

### 4. MACD Confirmation
- MACD histogram turning positive after being negative: bullish momentum
- MACD signal line crossover on D1: meaningful for swing
- Use as confirmation, not primary signal

## Horizon Estimation

| Setup Quality | Horizon |
|--------------|---------|
| RSI extreme + high volume + EMA cross | 15–20 days |
| RSI extreme + EMA alignment | 10–15 days |
| Single indicator (RSI only) | 5–8 days max |
| Unclear/mixed signals | DO NOT ENTER |

## Invalidation Conditions

Every swing thesis MUST have a concrete invalidation condition. Examples:
- "If daily close falls below EMA50" → structural support broken
- "If RSI recrosses 50 from oversold without a move higher" → momentum failed
- "If price closes more than 2×ATR against position" → move invalidated
- "If sentiment turns sharply negative within 3 days" → catalyst reversed

Vague invalidation ("if the trade goes against me") is NOT acceptable.

## Stop Loss Sizing

For swing trades, use ATR-based stops:
- Tight setup (high conviction): 1.0–1.5× ATR(14) on D1
- Medium setup: 1.5–2.0× ATR(14) on D1
- Never use a stop tighter than 1×ATR — this gets stopped by normal volatility

## What to AVOID

- Entering into a multi-week downtrend (falling knives): wait for structure
- Buying in overbought RSI for "breakout" — these rarely hold for 5+ days
- Holding past 20 days regardless of thesis: the 20-day hard limit is absolute
- Entering on thin volume: volume below 0.8× average is a red flag
- Multiple entries in correlated assets (e.g., NVDA + AMD + INTC simultaneously)

## Output Example (Strong Setup)

```json
{
  "symbol": "AAPL",
  "action": "buy",
  "confidence": 0.78,
  "reasoning": "RSI(14) D1 at 28.3 — most oversold since Oct 2022. EMA20 still above EMA50 (uptrend intact). Today's volume was 2.1× average — institutional buying on the dip. Finnhub sentiment neutral-to-positive (no negative news catalysts). Setup: RSI mean reversion within uptrend.",
  "signals_used": ["indicators_full_analysis", "finnhub_get_quote"],
  "suggested_stop_loss_atr_multiple": 1.5,
  "horizon_days": 12,
  "invalidation_condition": "Daily close below EMA50 ($188.20) or RSI rebounds above 50 without price recovering above $192"
}
```

## Output Example (Weak Setup — HOLD)

```json
{
  "symbol": "TSLA",
  "action": "hold",
  "confidence": 0.35,
  "reasoning": "RSI at 42 (neutral, no extreme). EMA20 below EMA50 (downtrend). Reddit sentiment slightly negative. No clear entry signal. Could go either way over 5-20 days.",
  "signals_used": ["indicators_full_analysis", "reddit_get_subreddit_sentiment"],
  "suggested_stop_loss_atr_multiple": 1.5,
  "horizon_days": 10,
  "invalidation_condition": "N/A — no trade entered"
}
```
