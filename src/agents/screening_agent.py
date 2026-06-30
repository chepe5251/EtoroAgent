"""
ScreeningAgent — two-stage funnel over the full symbol universe.

Stage 1a (deterministic): pandas-ta indicators on daily candles.
  Filter criteria (any ONE passes):
    - RSI(14) < 35 (oversold) or > 65 (overbought)
    - EMA20/EMA50 crossover in the last 3 days
    - Relative volume > 1.5× 20-day average

Stage 1b (fast LLM): single LLM call per batch of 8 symbols, no tools.
  Asks the model to pick the top 3 most promising for deep research.
  Falls back to passing all candidates if the LLM call fails.

Output: shortlist of up to 15 symbols sent to ResearchAgent for deep ReAct.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

try:
    import pandas as pd
    import pandas_ta as ta  # type: ignore
    _HAS_TA = True
except ImportError:
    _HAS_TA = False

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from src.core.etoro_client import EtoroClient

logger = logging.getLogger(__name__)

_BATCH_SIZE = 8            # symbols per LLM batch
_MAX_SHORTLIST = 15        # hard cap on output symbols
_CANDLE_COUNT = 60         # how many daily candles to fetch per symbol
_FETCH_BATCH = 10          # parallel HTTP requests per tick
_FETCH_DELAY = 1.5         # seconds between fetch batches


@dataclass
class ScreeningResult:
    symbol: str
    rsi: Optional[float] = None
    ema20: Optional[float] = None
    ema50: Optional[float] = None
    atr: Optional[float] = None
    rel_volume: Optional[float] = None
    ema_cross: bool = False
    score_tags: list[str] = field(default_factory=list)


class ScreeningAgent:
    def __init__(self, client: "EtoroClient"):
        self.client = client
        self._llm = AsyncOpenAI(
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("LLM_API_KEY", "lm-studio"),
        )
        self._screening_model = os.getenv(
            "SCREENING_LLM_MODEL",
            os.getenv("LLM_MODEL", "deepseek-coder-v2-lite-instruct"),
        )
        self._rsi_lo = float(os.getenv("SCREEN_RSI_OVERSOLD", "35"))
        self._rsi_hi = float(os.getenv("SCREEN_RSI_OVERBOUGHT", "65"))
        self._rel_vol = float(os.getenv("SCREEN_REL_VOL", "1.5"))
        self._ema_cross_days = int(os.getenv("SCREEN_EMA_CROSS_DAYS", "3"))

    # ── Public ────────────────────────────────────────────────────────────────

    async def run(self, symbols: list[str]) -> list[str]:
        """
        Run the two-stage funnel.
        Returns up to _MAX_SHORTLIST symbols for deep research.
        """
        logger.info("Screening: %d symbols → funnel start", len(symbols))

        candidates = await self._stage1a(symbols)
        logger.info(
            "Screening 1a: %d/%d passed deterministic filter",
            len(candidates), len(symbols),
        )

        if not candidates:
            return []

        shortlist = await self._stage1b(candidates)
        result = [r.symbol for r in shortlist[:_MAX_SHORTLIST]]
        logger.info("Screening 1b: shortlist=%s", result)
        return result

    # ── Stage 1a — deterministic ──────────────────────────────────────────────

    async def _stage1a(self, symbols: list[str]) -> list[ScreeningResult]:
        candles_map = await self._fetch_all_candles(symbols)
        candidates: list[ScreeningResult] = []
        for sym in symbols:
            candles = candles_map.get(sym, [])
            result = self._compute(sym, candles)
            if result and result.score_tags:
                candidates.append(result)
        return candidates

    async def _fetch_all_candles(self, symbols: list[str]) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for i in range(0, len(symbols), _FETCH_BATCH):
            batch = symbols[i : i + _FETCH_BATCH]
            tasks = [self.client.get_candles(sym, interval="D1", count=_CANDLE_COUNT) for sym in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for sym, res in zip(batch, results):
                if isinstance(res, Exception):
                    logger.debug("Candle fetch failed %s: %s", sym, res)
                    out[sym] = []
                else:
                    out[sym] = res
            if i + _FETCH_BATCH < len(symbols):
                await asyncio.sleep(_FETCH_DELAY)
        return out

    def _compute(self, symbol: str, candles: list[dict]) -> Optional[ScreeningResult]:
        if not _HAS_TA:
            # Without pandas-ta, pass everything through to LLM stage
            return ScreeningResult(symbol=symbol, score_tags=["no_ta_lib"])

        if len(candles) < 30:
            return None

        try:
            df = pd.DataFrame([{
                "open":   float(c.get("open",   c.get("o", 0))),
                "high":   float(c.get("high",   c.get("h", 0))),
                "low":    float(c.get("low",    c.get("l", 0))),
                "close":  float(c.get("close",  c.get("c", 0))),
                "volume": float(c.get("volume", c.get("v", 0))),
            } for c in candles])

            rsi_s   = ta.rsi(df["close"], length=14)
            ema20_s = ta.ema(df["close"], length=20)
            ema50_s = ta.ema(df["close"], length=50)
            atr_s   = ta.atr(df["high"], df["low"], df["close"], length=14)

            df["vol20"] = df["volume"].rolling(20).mean()
            df["rel_vol"] = df["volume"] / df["vol20"].replace(0, float("nan"))

            last_rsi    = float(rsi_s.iloc[-1])   if rsi_s.notna().any()   else None
            last_ema20  = float(ema20_s.iloc[-1]) if ema20_s.notna().any() else None
            last_ema50  = float(ema50_s.iloc[-1]) if ema50_s.notna().any() else None
            last_atr    = float(atr_s.iloc[-1])   if atr_s.notna().any()   else None
            last_rvol   = float(df["rel_vol"].iloc[-1]) if pd.notna(df["rel_vol"].iloc[-1]) else None

            # EMA cross detection over last N candles
            ema_cross = False
            for i in range(-self._ema_cross_days, 0):
                try:
                    prev_e20 = float(ema20_s.iloc[i - 1])
                    prev_e50 = float(ema50_s.iloc[i - 1])
                    curr_e20 = float(ema20_s.iloc[i])
                    curr_e50 = float(ema50_s.iloc[i])
                    if (prev_e20 < prev_e50 and curr_e20 >= curr_e50) or \
                       (prev_e20 > prev_e50 and curr_e20 <= curr_e50):
                        ema_cross = True
                        break
                except (IndexError, ValueError):
                    pass

            tags: list[str] = []
            if last_rsi is not None:
                if last_rsi < self._rsi_lo:
                    tags.append("rsi_oversold")
                elif last_rsi > self._rsi_hi:
                    tags.append("rsi_overbought")
            if ema_cross:
                tags.append("ema_cross")
            if last_rvol and last_rvol > self._rel_vol:
                tags.append("high_volume")

            return ScreeningResult(
                symbol=symbol,
                rsi=last_rsi,
                ema20=last_ema20,
                ema50=last_ema50,
                atr=last_atr,
                rel_volume=last_rvol,
                ema_cross=ema_cross,
                score_tags=tags,
            )
        except Exception as exc:
            logger.warning("Indicator compute failed for %s: %s", symbol, exc)
            return None

    # ── Stage 1b — LLM quick rank ─────────────────────────────────────────────

    async def _stage1b(self, candidates: list[ScreeningResult]) -> list[ScreeningResult]:
        if not candidates:
            return []
        shortlist: list[ScreeningResult] = []
        for i in range(0, len(candidates), _BATCH_SIZE):
            batch = candidates[i : i + _BATCH_SIZE]
            ranked = await self._rank_batch(batch)
            shortlist.extend(ranked)
        return shortlist

    async def _rank_batch(self, batch: list[ScreeningResult]) -> list[ScreeningResult]:
        """Single LLM call to pick top 3 from a batch — no tool calling."""

        def fmt(r: ScreeningResult) -> str:
            rsi_s  = f"{r.rsi:.1f}"    if r.rsi        is not None else "N/A"
            rvol_s = f"{r.rel_volume:.2f}" if r.rel_volume is not None else "N/A"
            ema_rel = "above" if (r.ema20 and r.ema50 and r.ema20 > r.ema50) else "below"
            return (
                f"{r.symbol}: RSI={rsi_s}, EMA20={ema_rel} EMA50, "
                f"RelVol={rvol_s}x, Flags={r.score_tags}"
            )

        lines = "\n".join(fmt(r) for r in batch)
        prompt = (
            f"You are a swing-trading screener (5-20 day horizon). "
            f"Select the TOP 3 symbols below most likely to move significantly in the next 5-20 days. "
            f"Prefer: clear RSI extremes with volume, fresh EMA crossovers. "
            f"Avoid: ambiguous signals or low relative volume.\n\n"
            f"Candidates:\n{lines}\n\n"
            f"Return ONLY valid JSON array (no markdown, no explanation):\n"
            f'[{{"symbol": "TICKER", "rank": 1, "reason": "brief"}}]'
        )
        try:
            resp = await self._llm.chat.completions.create(
                model=self._screening_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            if "```" in raw:
                parts = raw.split("```")
                for p in parts:
                    p = p.strip().lstrip("json").strip()
                    if p.startswith("["):
                        raw = p
                        break
            picks = json.loads(raw)
            selected = {item["symbol"] for item in picks}
            result = [r for r in batch if r.symbol in selected]
            logger.debug("LLM screening: %s → %s", [r.symbol for r in batch], list(selected))
            return result
        except Exception as exc:
            logger.warning("LLM screening batch failed (%s) — passing all %d candidates", exc, len(batch))
            return batch
