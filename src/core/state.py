from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional


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
        return (datetime.now(timezone.utc) - self.opened_at).days

    @property
    def max_hold_until(self) -> datetime:
        return self.opened_at + timedelta(days=self.horizon_days)

    @property
    def is_past_hard_exit(self) -> bool:
        return self.days_open >= 20


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
