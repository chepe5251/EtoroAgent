"""
RiskGate — 100% deterministic validation of a TradingThesis.
No LLM. No opinions. Hard rules only.

The LLM cannot override, argue with, or bypass these checks.
If validate() returns False, the trade does not happen. Period.
"""
from __future__ import annotations

import logging
import os

from src.core.state import ProjectState
from src.core.thesis import TradingThesis

logger = logging.getLogger(__name__)

_MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
_DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "3.0"))
_MIN_CONFIDENCE = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.65"))
_MIN_SIGNALS = int(os.getenv("MIN_SIGNALS_REQUIRED", "2"))
_MIN_REASONING_LEN = 50
_SWING_MIN_HORIZON = int(os.getenv("SWING_MIN_HORIZON_DAYS", "5"))
_SWING_MAX_HORIZON = int(os.getenv("SWING_MAX_HORIZON_DAYS", "20"))


def validate(
    thesis: TradingThesis,
    state: ProjectState,
    balance: float = 0.0,
    unrealized_pnl: float = 0.0,
) -> tuple[bool, str]:
    """
    Validate a trading thesis against deterministic risk rules.

    Args:
        thesis: The thesis produced by ResearchAgent
        state: Current system state (positions, P&L, block status)
        balance: Current available balance (used for daily loss % check)

    Returns:
        (allowed: bool, reason: str)
        If allowed is False, reason explains which rule failed.
    """
    # ── Daily reset (idempotent, safe to call here) ───────────────────
    state.reset_daily_if_needed()

    # Rule 0: hold actions never need validation
    if thesis.action == "hold":
        return False, "action is hold — no trade needed"

    # Rule 1: minimum confidence
    if thesis.confidence < _MIN_CONFIDENCE:
        return (
            False,
            f"confidence {thesis.confidence:.0%} below minimum {_MIN_CONFIDENCE:.0%}",
        )

    # Rule 2: minimum independent signals
    if len(thesis.signals_used) < _MIN_SIGNALS:
        return (
            False,
            f"only {len(thesis.signals_used)} signal(s) — need at least {_MIN_SIGNALS}",
        )

    # Rule 3: reasoning must be substantive
    if len(thesis.reasoning.strip()) < _MIN_REASONING_LEN:
        return (
            False,
            f"reasoning too short ({len(thesis.reasoning)} chars) — LLM must justify",
        )

    # Rule 4: daily loss block
    if state.is_risk_blocked:
        return False, f"daily loss block active: {state.risk_block_reason}"

    # Rule 5: check daily loss limit (proactive — block before it triggers).
    # Include unrealized losses (but not gains) for a conservative risk assessment.
    if balance > 0:
        effective_loss = -state.daily_pnl + max(0.0, -unrealized_pnl)
        loss_pct = (effective_loss / balance) * 100
        if loss_pct >= _DAILY_LOSS_LIMIT_PCT:
            reason = (
                f"daily loss {loss_pct:.2f}% >= limit {_DAILY_LOSS_LIMIT_PCT:.1f}% "
                f"(realized={state.daily_pnl:+.2f}, unrealized={unrealized_pnl:+.2f})"
            )
            state.block(reason)
            return False, reason

    # Rule 6: maximum simultaneous open positions
    if len(state.open_positions) >= _MAX_OPEN_POSITIONS:
        return (
            False,
            f"max open positions ({_MAX_OPEN_POSITIONS}) already reached",
        )

    # Rule 7: no duplicate symbol
    if any(p.symbol == thesis.symbol for p in state.open_positions):
        return False, f"position already open for {thesis.symbol}"

    # Rule 8: swing horizon must be in [5, 20] days
    if not (_SWING_MIN_HORIZON <= thesis.horizon_days <= _SWING_MAX_HORIZON):
        return (
            False,
            f"horizon_days={thesis.horizon_days} out of range "
            f"[{_SWING_MIN_HORIZON}, {_SWING_MAX_HORIZON}]",
        )

    logger.info(
        "RiskGate: APPROVED %s %s confidence=%.0f%% signals=%s",
        thesis.symbol, thesis.action.upper(), thesis.confidence * 100, thesis.signals_used,
    )
    return True, "approved"


def record_closed_pnl(state: ProjectState, pnl: float):
    """Update daily P&L after a position closes."""
    state.daily_pnl += pnl
    logger.info(
        "RiskGate: trade closed pnl=%.2f daily_pnl=%.2f", pnl, state.daily_pnl
    )
    # Check if this pushes us over the daily limit
    # (will be caught on the next validate() call; we block proactively here too
    #  if possible, but we don't have balance available — orchestrator will handle)
