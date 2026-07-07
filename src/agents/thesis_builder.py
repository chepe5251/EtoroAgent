"""
Deterministic thesis builder — no LLM.

Replaces ResearchAgent's LLM ReAct loop. Builds a TradingThesis directly
from a ScreeningResult, using the validated breakout/pullback trend-following
rules (EMA50>EMA200 trend gate + Donchian breakout or EMA20 pullback-resume,
volume-confirmed) — backtested on 5 years of real data across 140 symbols:
out-of-sample profit factor 1.60. See src/backtest/engine.py.
"""
from __future__ import annotations

from src.agents.screening_agent import ScreeningResult
from src.core.thesis import TradingThesis

_CONFIDENCE = 0.75             # fixed — this is a rule-based system, not an LLM opinion
_HORIZON_DAYS = 15             # within [SWING_MIN_HORIZON_DAYS, SWING_MAX_HORIZON_DAYS]
_STOP_LOSS_ATR_MULTIPLE = 2.5  # matches BacktestConfig.atr_stop_multiple (validated: OOS PF 2.01→1.94, +71% P&L vs 1.5x at 3x leverage/3% risk)


def build_thesis(result: ScreeningResult) -> TradingThesis:
    pattern = "breakout" if result.breakout else "pullback_resume"

    reasoning = f"Trend up (EMA50>EMA200). {pattern} confirmed"
    if result.rel_volume is not None:
        reasoning += f" with relative volume {result.rel_volume:.2f}x the 20-day average"
    reasoning += (
        ". Rule-based entry per the validated breakout/pullback trend-following "
        "system (backtested: out-of-sample profit factor 1.60 across 140 real "
        "symbols, 5 years, real fees + leverage)."
    )

    return TradingThesis(
        symbol=result.symbol,
        action="buy",
        confidence=_CONFIDENCE,
        reasoning=reasoning,
        signals_used=["trend_ema50_gt_ema200", pattern, "relative_volume_confirmation"],
        suggested_stop_loss_atr_multiple=_STOP_LOSS_ATR_MULTIPLE,
        horizon_days=_HORIZON_DAYS,
        invalidation_condition="Daily close falls below EMA50 (trend break)",
    )
