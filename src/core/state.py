from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_STATE_FILE = Path("state.json")


def get_hard_exit_days() -> int:
    return int(os.getenv("SWING_HARD_EXIT_DAYS", "20"))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Position:
    position_id: str
    instrument_id: str
    symbol: str
    is_buy: bool
    amount_usd: float
    entry_rate: float
    stop_loss_pct: float
    opened_at: datetime
    current_rate: float = 0.0
    atr: float = 0.0
    horizon_days: int = 10
    invalidation_condition: str = ""

    @property
    def days_open(self) -> int:
        return (_utcnow() - self.opened_at).days

    @property
    def max_hold_until(self) -> datetime:
        return self.opened_at + timedelta(days=self.horizon_days)

    @property
    def is_past_hard_exit(self) -> bool:
        return self.days_open >= get_hard_exit_days()

    def to_dict(self) -> dict:
        return {
            "position_id": self.position_id,
            "instrument_id": self.instrument_id,
            "symbol": self.symbol,
            "is_buy": self.is_buy,
            "amount_usd": self.amount_usd,
            "entry_rate": self.entry_rate,
            "stop_loss_pct": self.stop_loss_pct,
            "opened_at": self.opened_at.isoformat(),
            "current_rate": self.current_rate,
            "atr": self.atr,
            "horizon_days": self.horizon_days,
            "invalidation_condition": self.invalidation_condition,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        opened_at = datetime.fromisoformat(d["opened_at"])
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        return cls(
            position_id=d["position_id"],
            instrument_id=d["instrument_id"],
            symbol=d["symbol"],
            is_buy=bool(d["is_buy"]),
            amount_usd=float(d["amount_usd"]),
            entry_rate=float(d["entry_rate"]),
            stop_loss_pct=float(d["stop_loss_pct"]),
            opened_at=opened_at,
            current_rate=float(d.get("current_rate", 0.0)),
            atr=float(d.get("atr", 0.0)),
            horizon_days=int(d.get("horizon_days", 10)),
            invalidation_condition=str(d.get("invalidation_condition", "")),
        )


class ProjectState:
    def __init__(self):
        self.open_positions: list[Position] = []
        self.daily_pnl: float = 0.0
        self.is_risk_blocked: bool = False
        self.risk_block_reason: str = ""
        self.last_updated: datetime = _utcnow()
        self._daily_reset_date: str = _utcnow().date().isoformat()

    def reset_daily_if_needed(self) -> bool:
        today = _utcnow().date().isoformat()
        if today != self._daily_reset_date:
            self.daily_pnl = 0.0
            self.is_risk_blocked = False
            self.risk_block_reason = ""
            self._daily_reset_date = today
            self.save()
            return True
        return False

    def get_open_position(self, symbol: str) -> Optional[Position]:
        for pos in self.open_positions:
            if pos.symbol == symbol:
                return pos
        return None

    def remove_position(self, position_id: str):
        self.open_positions = [p for p in self.open_positions if p.position_id != position_id]

    def block(self, reason: str):
        self.is_risk_blocked = True
        self.risk_block_reason = reason
        self.save()

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> None:
        """Persist state to JSON for crash recovery."""
        target = path or _STATE_FILE
        data = {
            "open_positions": [p.to_dict() for p in self.open_positions],
            "daily_pnl": self.daily_pnl,
            "is_risk_blocked": self.is_risk_blocked,
            "risk_block_reason": self.risk_block_reason,
            "_daily_reset_date": self._daily_reset_date,
            "saved_at": _utcnow().isoformat(),
        }
        try:
            target.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.warning("Could not save state to %s: %s", target, exc)

    @classmethod
    def load(cls, path: Path | None = None) -> "ProjectState":
        """Load persisted state. Returns a fresh state if file is missing or corrupt."""
        target = path or _STATE_FILE
        state = cls()
        if not target.exists():
            return state
        try:
            data = json.loads(target.read_text())
            state.open_positions = [
                Position.from_dict(p) for p in data.get("open_positions", [])
            ]
            state.daily_pnl = float(data.get("daily_pnl", 0.0))
            state.is_risk_blocked = bool(data.get("is_risk_blocked", False))
            state.risk_block_reason = str(data.get("risk_block_reason", ""))
            state._daily_reset_date = str(
                data.get("_daily_reset_date", _utcnow().date().isoformat())
            )
            logger.info(
                "State loaded from %s: %d open positions, daily_pnl=%.2f",
                target, len(state.open_positions), state.daily_pnl,
            )
        except Exception as exc:
            logger.warning("Could not load state from %s: %s — using fresh state", target, exc)
        return state
