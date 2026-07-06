"""
Append-only structured trade history — one JSON object per line.

Separate from ProjectState (which only holds currently-open positions):
this file is never pruned, so the dashboard can build an equity curve
and win-rate stats across the bot's whole lifetime.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_TRADE_LOG_FILE = Path(__file__).parent.parent.parent / "trades.jsonl"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_open(
    position_id: str,
    symbol: str,
    is_buy: bool,
    amount_usd: float,
    entry_rate: float,
    stop_loss_pct: float,
    horizon_days: int,
) -> None:
    _append({
        "event": "open",
        "timestamp": _utcnow_iso(),
        "position_id": position_id,
        "symbol": symbol,
        "is_buy": is_buy,
        "amount_usd": amount_usd,
        "entry_rate": entry_rate,
        "stop_loss_pct": stop_loss_pct,
        "horizon_days": horizon_days,
    })


def log_close(
    position_id: str,
    symbol: str,
    is_buy: bool,
    amount_usd: float,
    entry_rate: float,
    close_rate: float,
    pnl: float,
    duration_hours: float,
    reason: str,
) -> None:
    _append({
        "event": "close",
        "timestamp": _utcnow_iso(),
        "position_id": position_id,
        "symbol": symbol,
        "is_buy": is_buy,
        "amount_usd": amount_usd,
        "entry_rate": entry_rate,
        "close_rate": close_rate,
        "pnl": pnl,
        "duration_hours": duration_hours,
        "reason": reason,
    })


def _append(record: dict, path: Path | None = None) -> None:
    target = path or _TRADE_LOG_FILE
    try:
        with target.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("Could not append to trade log %s: %s", target, exc)


def read_all(path: Path | None = None) -> list[dict]:
    """Read the full trade history. Returns [] if the file doesn't exist yet."""
    target = path or _TRADE_LOG_FILE
    if not target.exists():
        return []
    records = []
    for line in target.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed trade log line: %s", line[:100])
    return records
