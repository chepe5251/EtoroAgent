"""
PositionReviewAgent — daily review of open swing positions.

For each open position it:
  1. Enforces the 20-day hard exit (deterministic, no LLM).
  2. Runs a short ReAct loop to evaluate if the original thesis still holds.
  3. Acts on the LLM verdict: EXIT → close, TIGHTEN_STOP → adjust, HOLD → skip.
"""
from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

from src.core.state import Position, ProjectState
from src.core.thesis import TradingThesis
from src.llm.react_runtime import ReActRuntime

if TYPE_CHECKING:
    from src.mcp_clients.mcp_manager import MCPManager
    from src.agents.execution_agent import ExecutionAgent
    from src.agents.notification_agent import NotificationAgent

logger = logging.getLogger(__name__)

_HARD_EXIT_DAYS = int(os.getenv("SWING_HARD_EXIT_DAYS", "20"))
_REVIEW_MIN_CONFIDENCE = float(os.getenv("REVIEW_MIN_CONFIDENCE", "0.55"))

_REVIEW_SYSTEM_PROMPT = """\
Sos un analista de seguimiento de posiciones swing. Tenés una posición abierta
y debés decidir si el trade sigue siendo válido.

REGLAS:
1. Solo usás herramientas de lectura (indicadores, sentiment). No podés operar.
2. Verificá si la condición de invalidación original ya se cumplió.
3. Evaluá si los indicadores confirman o contradicen la tesis original.
4. Tu respuesta FINAL debe ser este JSON exacto:
{
  "action": "hold" | "exit" | "tighten_stop",
  "reason": "<explicación concisa>",
  "new_stop_atr_multiple": <float o null>
}

Si hay señales mixtas, preferí "hold" sobre "exit" para posiciones jóvenes (<5 días).
Si la condición de invalidación se cumplió claramente, devolvé "exit".
Si el trade está funcionando bien (precio avanzó en la dirección esperada), devolvé "tighten_stop".
"""


class PositionReviewAgent:
    """Reviews open positions daily and acts on stale or invalidated trades."""

    def __init__(
        self,
        mcp_manager: "MCPManager",
        execution_agent: "ExecutionAgent",
        notification_agent: "NotificationAgent",
        state: ProjectState,
    ):
        self.mcp_manager = mcp_manager
        self.execution_agent = execution_agent
        self.notification_agent = notification_agent
        self.state = state

    async def review_all(self):
        """Review every open position. Called once per day."""
        positions = list(self.state.open_positions)
        if not positions:
            logger.info("PositionReview: no open positions to review")
            return

        logger.info("PositionReview: reviewing %d open position(s)", len(positions))
        for pos in positions:
            try:
                await self._review_position(pos)
            except Exception as exc:
                logger.error("Error reviewing position %s: %s", pos.symbol, exc, exc_info=True)

    async def _review_position(self, pos: Position):
        logger.info(
            "PositionReview: %s — day %d/%d (horizon=%d)",
            pos.symbol, pos.days_open, _HARD_EXIT_DAYS, pos.horizon_days,
        )

        # ── Hard exit: 20-day absolute limit (deterministic) ──────────────
        if pos.is_past_hard_exit:
            logger.warning(
                "PositionReview: %s hit 20-day hard limit — forcing EXIT", pos.symbol
            )
            await self._close(pos, reason=f"20-day hard exit limit reached (opened {pos.days_open}d ago)")
            return

        # ── Soft exit: past the horizon the LLM chose ─────────────────────
        # Still let LLM decide, but note in prompt that target window passed
        past_horizon = pos.days_open >= pos.horizon_days

        # ── LLM review via short ReAct loop ───────────────────────────────
        verdict = await self._llm_review(pos, past_horizon=past_horizon)
        if verdict is None:
            logger.warning("PositionReview: %s — no LLM verdict, skipping", pos.symbol)
            return

        action = verdict.get("action", "hold").lower()
        reason = verdict.get("reason", "")
        new_stop = verdict.get("new_stop_atr_multiple")

        logger.info("PositionReview: %s → %s (%s)", pos.symbol, action.upper(), reason)

        if action == "exit":
            await self._close(pos, reason=f"LLM review: {reason}")
        elif action == "tighten_stop" and new_stop is not None:
            await self._tighten(pos, float(new_stop), reason)
        else:
            logger.info("PositionReview: %s → HOLD, no action", pos.symbol)

    async def _llm_review(self, pos: Position, past_horizon: bool) -> dict | None:
        horizon_note = (
            f"NOTA: ya pasaron {pos.days_open}d del horizonte estimado ({pos.horizon_days}d). "
            "Si no hay motivo claro para mantener, preferí 'exit'."
            if past_horizon
            else ""
        )
        user_prompt = (
            f"Revisá la posición abierta:\n"
            f"  Symbol: {pos.symbol}\n"
            f"  Dirección: {'BUY (largo)' if pos.is_buy else 'SELL (corto)'}\n"
            f"  Precio entrada: {pos.entry_rate}\n"
            f"  Días abierta: {pos.days_open}\n"
            f"  Horizonte objetivo: {pos.horizon_days} días\n"
            f"  Condición de invalidación: '{pos.invalidation_condition or 'no especificada'}'\n"
            f"  {horizon_note}\n\n"
            f"Usá las tools para ver el estado actual del mercado y determiná si "
            f"la condición de invalidación se cumplió o si el trade sigue válido. "
            f"Devolvé el JSON de revisión."
        )

        runtime = ReActRuntime(
            mcp_manager=self.mcp_manager,
            system_prompt=_REVIEW_SYSTEM_PROMPT,
            max_iterations=4,
        )
        result = await runtime.run(user_prompt)

        if result.get("error"):
            logger.warning("PositionReview LLM error for %s: %s", pos.symbol, result["error"])

        raw = result.get("thesis")
        if raw and isinstance(raw, dict) and "action" in raw:
            return raw

        logger.warning("PositionReview: %s — could not parse verdict from LLM", pos.symbol)
        return None

    async def _close(self, pos: Position, reason: str):
        """Close a position and notify."""
        try:
            await self.execution_agent.close_position(pos.position_id, reason=reason)
            logger.info("PositionReview: closed %s — %s", pos.symbol, reason)
        except Exception as exc:
            logger.error("PositionReview: close failed for %s: %s", pos.symbol, exc)

    async def _tighten(self, pos: Position, new_stop_multiple: float, reason: str):
        """Tighten stop loss on a position."""
        if pos.atr <= 0:
            logger.warning("PositionReview: cannot tighten stop for %s — ATR=0", pos.symbol)
            return
        new_stop_pct = (pos.atr * new_stop_multiple / pos.entry_rate) * 100
        logger.info(
            "PositionReview: tightening stop on %s → %.2f%% (ATR×%.1f): %s",
            pos.symbol, new_stop_pct, new_stop_multiple, reason,
        )
        # Update in memory; actual API update handled by trailing stop agent
        pos.stop_loss_pct = min(pos.stop_loss_pct, new_stop_pct)
