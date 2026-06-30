"""
NotificationAgent — sends Telegram messages for key trading events.
Deterministic — no LLM. Fire-and-forget; failures are logged but don't
crash the main loop.
"""
from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import TYPE_CHECKING

import httpx
from dotenv import load_dotenv

from src.core.state import Position, ProjectState
from src.core.thesis import TradingThesis

if TYPE_CHECKING:
    pass

load_dotenv()
logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org"


class NotificationAgent:
    def __init__(self, state: ProjectState):
        self.state = state
        self.token = os.getenv("TELEGRAM_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self.token and self.chat_id)
        if not self._enabled:
            logger.warning(
                "NotificationAgent: TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set — "
                "notifications disabled"
            )

    async def _send(self, text: str):
        if not self._enabled:
            logger.debug("NotificationAgent [disabled]: %s", text[:80])
            return
        url = f"{_TELEGRAM_API}/bot{self.token}/sendMessage"
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                resp = await client.post(
                    url,
                    json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
                )
                resp.raise_for_status()
            except Exception as exc:
                logger.warning("NotificationAgent: Telegram send failed: %s", exc)

    # ── Event handlers ─────────────────────────────────────────────────

    async def send_position_opened(self, pos: Position):
        direction = "🟢 LONG" if pos.is_buy else "🔴 SHORT"
        await self._send(
            f"<b>Position Opened</b> {direction}\n"
            f"Symbol: <code>{pos.symbol}</code>\n"
            f"Price: {pos.entry_rate}\n"
            f"Amount: ${pos.amount_usd:.2f}\n"
            f"Stop-loss: {pos.stop_loss_pct:.2f}%\n"
            f"ID: <code>{pos.position_id}</code>"
        )

    async def send_position_closed(self, pos: Position, pnl: float, duration: timedelta):
        emoji = "✅" if pnl >= 0 else "❌"
        h, rem = divmod(int(duration.total_seconds()), 3600)
        m = rem // 60
        await self._send(
            f"<b>Position Closed</b> {emoji}\n"
            f"Symbol: <code>{pos.symbol}</code>\n"
            f"P&amp;L: <b>${pnl:+.2f}</b>\n"
            f"Duration: {h}h {m}m\n"
            f"Entry: {pos.entry_rate} | Exit: {pos.current_rate}"
        )

    async def send_thesis_rejected(self, thesis: TradingThesis, reason: str):
        await self._send(
            f"⛔ <b>Thesis Rejected</b>\n"
            f"Symbol: <code>{thesis.symbol}</code>\n"
            f"Action: {thesis.action.upper()} (conf={thesis.confidence:.0%})\n"
            f"Reason: {reason}\n"
            f"Signals: {', '.join(thesis.signals_used) or 'none'}"
        )

    async def send_risk_blocked(self, reason: str):
        await self._send(
            f"⚠️ <b>Risk Limit Reached</b>\n"
            f"Reason: {reason}\n"
            f"Daily P&amp;L: <b>${self.state.daily_pnl:+.2f}</b>\n"
            f"No new trades until midnight UTC."
        )

    async def send_critical_error(self, message: str):
        await self._send(f"🚨 <b>Critical Error</b>\n<code>{message[:1000]}</code>")

    async def send_daily_summary(self):
        pnl = self.state.daily_pnl
        emoji = "✅" if pnl >= 0 else "❌"
        open_lines = ""
        if self.state.open_positions:
            open_lines = "\n<b>Open positions:</b>\n" + "\n".join(
                f"  • {p.symbol} {'LONG' if p.is_buy else 'SHORT'} ${p.amount_usd:.2f}"
                for p in self.state.open_positions
            )
        await self._send(
            f"📊 <b>Daily Summary</b>\n"
            f"P&amp;L: {emoji} <b>${pnl:+.2f}</b>\n"
            f"Risk blocked: {'Yes ⛔' if self.state.is_risk_blocked else 'No'}"
            f"{open_lines}"
        )

    async def send_startup(self, mode: str, symbols: list[str]):
        await self._send(
            f"🤖 <b>etoroAgent Started</b>\n"
            f"Mode: <code>{mode.upper()}</code>\n"
            f"Watching: {', '.join(symbols)}\n"
            f"Engine: ReAct + Qwen2.5-7B-Instruct (LM Studio)"
        )
