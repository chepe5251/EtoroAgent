import json
import logging
import os
from typing import Optional

import litellm
from dotenv import load_dotenv

from src.core.state import ProjectState, Signal

load_dotenv()
logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.65

_SIGNAL_PROMPT = """\
You are an expert algorithmic trader. Analyze the market data below and decide \
whether to BUY, SELL, or HOLD each symbol. Consider technical indicators, \
sentiment, and existing positions to choose the best strategy \
(technical-only, sentiment-only, or mixed).

## Current balance available: ${balance:.2f}

## Open positions:
{open_positions}

## Market data per symbol:
{market_data}

Return ONLY a valid JSON array (no markdown, no extra text) with this schema:
[
  {{
    "symbol": "BTC",
    "action": "BUY",
    "confidence": 0.80,
    "reasoning": "RSI oversold + positive sentiment + MACD crossover"
  }}
]
Each action must be exactly BUY, SELL, or HOLD.
Confidence must be a float between 0.0 and 1.0.
Include all symbols from the input, even if action is HOLD.
"""


class SignalAgent:
    """Uses an LLM to generate BUY/SELL/HOLD signals from market data and sentiment."""

    def __init__(self, state: ProjectState):
        self.state = state
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o")
        self.llm_base_url: Optional[str] = os.getenv("LLM_BASE_URL") or None
        self.llm_api_key: Optional[str] = os.getenv("LLM_API_KEY") or None

    async def run(self, balance: float) -> list[Signal]:
        if not self.state.market_data:
            logger.warning("SignalAgent: no market data available")
            return []

        market_block = self._format_market_data()
        positions_block = self._format_positions()

        prompt = _SIGNAL_PROMPT.format(
            balance=balance,
            open_positions=positions_block or "None",
            market_data=market_block,
        )

        kwargs: dict = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        if self.llm_base_url:
            kwargs["base_url"] = self.llm_base_url
        if self.llm_api_key:
            kwargs["api_key"] = self.llm_api_key

        try:
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            raw_signals: list[dict] = json.loads(content)
        except Exception as exc:
            logger.error("SignalAgent LLM call failed: %s", exc, exc_info=True)
            return []

        signals: list[Signal] = []
        for item in raw_signals:
            try:
                confidence = float(item.get("confidence", 0))
                action = str(item.get("action", "HOLD")).upper()
                if action not in ("BUY", "SELL", "HOLD"):
                    action = "HOLD"
                if confidence < MIN_CONFIDENCE:
                    logger.info(
                        "SignalAgent: %s %s confidence=%.2f below threshold — skipping",
                        item.get("symbol"), action, confidence,
                    )
                    continue
                signals.append(
                    Signal(
                        symbol=item["symbol"],
                        action=action,
                        confidence=confidence,
                        reasoning=item.get("reasoning", ""),
                    )
                )
            except (KeyError, ValueError) as exc:
                logger.warning("SignalAgent: malformed signal item %s: %s", item, exc)

        self.state.signals = signals
        logger.info(
            "SignalAgent: generated %d actionable signals (above %.0f%% confidence)",
            len(signals), MIN_CONFIDENCE * 100,
        )
        return signals

    def _format_market_data(self) -> str:
        lines = []
        for symbol, data in self.state.market_data.items():
            ind = data.get("indicators", {})
            sentiment = self.state.sentiment.get(symbol, 0.0)
            macd_d = ind.get("macd") or {}
            bb_d = ind.get("bollinger") or {}
            lines.append(
                f"### {symbol}\n"
                f"  close={ind.get('last_close')}\n"
                f"  RSI(14)={ind.get('rsi_14')}\n"
                f"  MACD line={macd_d.get('macd_line')} signal={macd_d.get('signal_line')} hist={macd_d.get('histogram')}\n"
                f"  EMA20={ind.get('ema_20')} EMA50={ind.get('ema_50')}\n"
                f"  BB upper={bb_d.get('upper')} middle={bb_d.get('middle')} lower={bb_d.get('lower')}\n"
                f"  ATR(14)={ind.get('atr_14')}\n"
                f"  RelVol={ind.get('relative_volume')}\n"
                f"  Sentiment={sentiment:.2f}"
            )
        return "\n".join(lines)

    def _format_positions(self) -> str:
        if not self.state.open_positions:
            return ""
        lines = []
        for p in self.state.open_positions:
            direction = "LONG" if p.is_buy else "SHORT"
            pnl_pct = 0.0
            if p.entry_rate and p.current_rate:
                if p.is_buy:
                    pnl_pct = (p.current_rate - p.entry_rate) / p.entry_rate * 100
                else:
                    pnl_pct = (p.entry_rate - p.current_rate) / p.entry_rate * 100
            lines.append(
                f"  {p.symbol} {direction} ${p.amount_usd:.2f} "
                f"entry={p.entry_rate} current={p.current_rate} pnl={pnl_pct:.2f}%"
            )
        return "\n".join(lines)
