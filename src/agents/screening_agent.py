"""
ScreeningAgent — 100% deterministic filter over the full symbol universe.
No LLM anywhere in this path (see src/agents/thesis_builder.py for how the
resulting candidates become an order).

Pure-Python indicators on daily candles. Trend-following filter, validated
against 5 years of real eToro data with full fees + leverage in
src/backtest (see src/backtest/engine.py's use_breakout_signal /
use_pullback_signal / trend_filter_type="ema50_200"):
  - Trend gate: EMA50 > EMA200 (must hold for ANY signal below to count)
  - Entry A: Donchian breakout — close > highest high of prior 20 bars
  - Entry B: EMA20 pullback-resume — close crosses back above EMA20
  - Both require relative volume > 1.5x the 20-day average (equities only;
    eToro reports no crypto volume, so crypto is excluded from this gate
    entirely upstream via WATCH_REGIONS)

Output: up to _MAX_SHORTLIST candidates, in universe order, sent straight to
build_thesis() and the deterministic risk gate.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from src.tools.technical import rsi as _rsi, ema as _ema, atr as _atr, relative_volume as _rvol

if TYPE_CHECKING:
    from src.core.etoro_client import EtoroClient

logger = logging.getLogger(__name__)

_MAX_SHORTLIST = 15        # hard cap on output symbols
_CANDLE_COUNT = 250        # need >=200 bars for a stable EMA200 trend gate
_FETCH_BATCH = 10          # parallel HTTP requests per tick
_FETCH_DELAY = 1.5         # seconds between fetch batches


@dataclass
class ScreeningResult:
    symbol: str
    rsi: Optional[float] = None
    ema20: Optional[float] = None
    ema50: Optional[float] = None
    ema200: Optional[float] = None
    atr: Optional[float] = None
    rel_volume: Optional[float] = None
    trend_up: bool = False
    breakout: bool = False
    pullback_resume: bool = False
    score_tags: list[str] = field(default_factory=list)


class ScreeningAgent:
    def __init__(self, client: "EtoroClient"):
        self.client = client
        self._rel_vol = float(os.getenv("SCREEN_REL_VOL", "1.5"))
        self._donchian_lookback = int(os.getenv("SCREEN_DONCHIAN_LOOKBACK", "20"))

    # ── Public ────────────────────────────────────────────────────────────────

    async def run(self, symbols: list[str]) -> list[ScreeningResult]:
        """
        Run the deterministic filter over the whole universe.
        Returns up to _MAX_SHORTLIST candidates, in universe order.
        """
        logger.info("Screening: %d symbols → funnel start", len(symbols))

        candidates = await self._stage1a(symbols)
        logger.info(
            "Screening: %d/%d passed deterministic filter",
            len(candidates), len(symbols),
        )
        result = candidates[:_MAX_SHORTLIST]
        logger.info("Screening: shortlist=%s", [r.symbol for r in result])
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
        if len(candles) < self._donchian_lookback + 2:
            return None

        try:
            closes  = [float(c.get("close")  or c.get("c") or 0) for c in candles]
            highs   = [float(c.get("high")   or c.get("h") or 0) for c in candles]
            lows    = [float(c.get("low")    or c.get("l") or 0) for c in candles]
            volumes = [float(c.get("volume") or c.get("v") or 0) for c in candles]

            last_rsi  = _rsi(closes, 14)
            last_atr  = _atr(highs, lows, closes, 14)
            last_rvol = _rvol(volumes, 20)

            ema20_series  = _ema(closes, 20)
            ema50_series  = _ema(closes, 50)
            ema200_series = _ema(closes, 200)

            last_ema20  = ema20_series[-1]  if ema20_series  else None
            last_ema50  = ema50_series[-1]  if ema50_series  else None
            last_ema200 = ema200_series[-1] if ema200_series else None

            # Trend gate: EMA50 > EMA200 (need real history for this to mean
            # anything — with < 200 candles this is left False, no signal).
            trend_up = (
                last_ema50 is not None and last_ema200 is not None
                and last_ema50 > last_ema200
            )

            # Volume confirmation — required for either entry below. eToro
            # reports no crypto volume (rel_vol always None there); crypto is
            # excluded from this universe upstream via WATCH_REGIONS.
            vol_ok = last_rvol is not None and last_rvol > self._rel_vol

            # Entry A: Donchian breakout — close clears the prior N-bar high
            # (excludes the current bar, no look-ahead).
            n = self._donchian_lookback
            donchian_high = max(highs[-(n + 1):-1]) if len(highs) > n else None
            breakout = (
                trend_up and vol_ok and donchian_high is not None
                and closes[-1] > donchian_high
            )

            # Entry B: EMA20 pullback-resume — close was at/below EMA20 the
            # prior bar and has now crossed back above it.
            pullback_resume = False
            if trend_up and vol_ok and len(ema20_series) >= 2 and len(closes) >= 2:
                prev_close, curr_close = closes[-2], closes[-1]
                prev_ema20, curr_ema20 = ema20_series[-2], ema20_series[-1]
                pullback_resume = prev_close <= prev_ema20 and curr_close > curr_ema20

            tags: list[str] = []
            if breakout:
                tags.append("breakout")
            if pullback_resume:
                tags.append("pullback_resume")

            return ScreeningResult(
                symbol=symbol,
                rsi=last_rsi,
                ema20=last_ema20,
                ema50=last_ema50,
                ema200=last_ema200,
                atr=last_atr,
                rel_volume=last_rvol,
                trend_up=trend_up,
                breakout=breakout,
                pullback_resume=pullback_resume,
                score_tags=tags,
            )
        except Exception as exc:
            logger.warning("Indicator compute failed for %s: %s", symbol, exc)
            return None
