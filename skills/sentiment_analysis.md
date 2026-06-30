# Sentiment Analysis Criteria

## Source Reliability Hierarchy
1. **Finnhub news sentiment** (highest reliability): Professional news sources, structured data, less noise.
2. **CryptoPanic** (medium reliability for crypto): Community-curated, filters help (use "bullish"/"bearish" for structured signal).
3. **Reddit** (lowest reliability, earliest signal): High noise, but often the first to react to developments. Use as a contrarian indicator when sentiment is extreme.

## Interpreting Finnhub News Sentiment
- `bullishPercent > 0.65`: Clearly positive news flow.
- `bullishPercent < 0.35`: Clearly negative news flow.
- Between 0.35–0.65: Ambiguous — don't count this as a signal.
- `companyNewsScore` above 0: Net positive. Below 0: Net negative.
- Look at `articlesInLastWeek` — if < 5, there's not enough news to be reliable.

## Interpreting CryptoPanic
- `filter: "bullish"` vs `filter: "bearish"` — compare post counts. If bullish_count / bearish_count > 2, that's a clear bullish signal.
- `filter: "hot"` — shows what the community is most engaged with. High engagement without clear direction = noise, not a signal.
- `panic` votes > `positive` votes: Fear signal — often contrarian bullish at extreme levels for crypto.

## Interpreting Reddit Sentiment
- `sentiment_hint: "bullish"` with `posts_found > 10` and `avg_score > 200`: Meaningful bullish signal.
- `sentiment_hint: "bearish"` with same criteria: Meaningful bearish signal.
- Extreme euphoria (`avg_score > 1000`, `bullish_posts >> bearish_posts`): **CONTRARIAN WARNING** — markets often reverse at peak Reddit enthusiasm. Lower your confidence on BUY signals.
- Low post count (`posts_found < 5`): Ignore Reddit for this symbol this cycle.

## Weighting Rule
When combining sentiment sources:
- Finnhub: 40% weight (most structured, least noisy)
- CryptoPanic (crypto only): 35% weight
- Reddit: 25% weight (most noise, use as tiebreaker only)

If only one source is available, weight = 100% of that source, but flag confidence as "moderate at best" (max 0.70 confidence for single-source sentiment).

## Sentiment vs Technical: Who Wins?
- **Aligned**: Both point the same direction → strong combined signal. Weight confidence high.
- **Divergence**: Technical says BUY but sentiment is strongly negative (or vice versa) → this is a HOLD. Don't fight contradictory signals.
- **Extreme negative sentiment + technical oversold**: Possible capitulation bottom. Valid BUY thesis but only with high RSI/MACD confirmation.
