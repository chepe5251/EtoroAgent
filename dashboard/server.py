"""
Read-only status dashboard for the etoroAgent bot.

Binds to 127.0.0.1 only — access via SSH tunnel:
    ssh -L 8080:localhost:8080 deploy@<vps-ip>
then open http://localhost:8080 locally. Never expose this port publicly:
it shows live balance and position data.
"""
from __future__ import annotations

import csv
import io
import os
import sys
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from src.config.universe import get_symbols
from src.core import trade_log
from src.core.etoro_client import EtoroClient
from src.core.state import ProjectState

_ROOT = Path(__file__).parent.parent
_LOG_FILE = _ROOT / "logs" / "etoroAgent.log"

_client: EtoroClient | None = None

_WATCH_REGIONS = [r.strip().upper() for r in os.getenv("WATCH_REGIONS", "US,EU,ASIA,CRYPTO").split(",")]

# Mirrors Orchestrator._setup_schedules() in src/core/orchestrator.py — kept as a
# lightweight local copy so the dashboard doesn't need to import/start the live
# scheduler. Update both places together if the schedule ever changes.
_REGION_SCHEDULES = {
    "US":        dict(scan=(9, 15), execute=(9, 30), tz="America/New_York"),
    "EU":        dict(scan=(8, 45), execute=(9, 0), tz="Europe/Berlin"),
    "ASIA":      dict(scan=(9, 15), execute=(9, 30), tz="America/New_York"),
    "HONGKONG":  dict(scan=(9, 15), execute=(9, 30), tz="Asia/Hong_Kong"),
    "JAPAN":     dict(scan=(8, 45), execute=(9, 0), tz="Asia/Tokyo"),
    "GERMANY":     dict(scan=(8, 45), execute=(9, 0), tz="Europe/Berlin"),
    "FRANCE":      dict(scan=(8, 45), execute=(9, 0), tz="Europe/Paris"),
    "SWITZERLAND": dict(scan=(8, 45), execute=(9, 0), tz="Europe/Zurich"),
    "AUSTRALIA":   dict(scan=(9, 45), execute=(10, 0), tz="Australia/Sydney"),
    "SWEDEN":      dict(scan=(8, 45), execute=(9, 0), tz="Europe/Stockholm"),
    "UK":          dict(scan=(7, 45), execute=(8, 0), tz="Europe/London"),
}


def _next_occurrence(hour: int, minute: int, tz: str) -> str | None:
    zone = ZoneInfo(tz)
    now = datetime.now(zone)
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate.astimezone(ZoneInfo("UTC")).isoformat()


def _symbol_region_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for region in _WATCH_REGIONS:
        for symbol in get_symbols(region):
            mapping[symbol] = region
    return mapping


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = EtoroClient()
    async with _client:
        yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/status")
async def status():
    state = ProjectState.load()
    balance = 0.0
    portfolio_error = None
    try:
        balance = await _client.get_balance()
    except Exception as exc:
        portfolio_error = str(exc)

    positions = [
        {
            "position_id": p.position_id,
            "symbol": p.symbol,
            "is_buy": p.is_buy,
            "amount_usd": p.amount_usd,
            "entry_rate": p.entry_rate,
            "current_rate": p.current_rate,
            "stop_loss_pct": p.stop_loss_pct,
            "horizon_days": p.horizon_days,
            "days_open": p.days_open,
            "unrealized_pnl": (
                (p.current_rate - p.entry_rate) / p.entry_rate * p.amount_usd
                if p.is_buy and p.entry_rate else
                (p.entry_rate - p.current_rate) / p.entry_rate * p.amount_usd
                if p.entry_rate else 0.0
            ),
        }
        for p in state.open_positions
    ]

    return {
        "mode": os.getenv("ETORO_MODE", "demo").upper(),
        "balance": balance,
        "balance_error": portfolio_error,
        "daily_pnl": state.daily_pnl,
        "is_risk_blocked": state.is_risk_blocked,
        "risk_block_reason": state.risk_block_reason,
        "open_positions": positions,
    }


@app.get("/api/trades")
async def trades():
    records = trade_log.read_all()
    closes = [r for r in records if r.get("event") == "close"]
    opens = {r["position_id"]: r for r in records if r.get("event") == "open"}

    wins = sum(1 for c in closes if c.get("pnl", 0) > 0)
    total_pnl = sum(c.get("pnl", 0) for c in closes)

    cumulative = 0.0
    equity_curve = []
    for c in closes:
        cumulative += c.get("pnl", 0)
        equity_curve.append({"timestamp": c["timestamp"], "cumulative_pnl": round(cumulative, 2)})

    return {
        "closed_trades": list(reversed(closes)),
        "open_count_ever": len(opens),
        "closed_count": len(closes),
        "win_count": wins,
        "win_rate": (wins / len(closes) * 100) if closes else None,
        "total_realized_pnl": round(total_pnl, 2),
        "equity_curve": equity_curve,
    }


@app.get("/api/activity")
async def activity(lines: int = 40):
    if not _LOG_FILE.exists():
        return {"lines": []}
    tail = deque(maxlen=lines)
    with _LOG_FILE.open() as f:
        for line in f:
            tail.append(line.rstrip("\n"))
    return {"lines": list(tail)}


@app.get("/api/regions")
async def regions():
    state = ProjectState.load()
    sym_region = _symbol_region_map()

    records = trade_log.read_all()
    closes = [r for r in records if r.get("event") == "close"]

    open_by_region: dict[str, int] = {r: 0 for r in _WATCH_REGIONS}
    for p in state.open_positions:
        region = sym_region.get(p.symbol, "OTHER")
        open_by_region[region] = open_by_region.get(region, 0) + 1

    stats: dict[str, dict] = {}
    for region in _WATCH_REGIONS:
        stats[region] = {"trades": 0, "wins": 0, "pnl": 0.0}

    for c in closes:
        region = sym_region.get(c.get("symbol"), "OTHER")
        bucket = stats.setdefault(region, {"trades": 0, "wins": 0, "pnl": 0.0})
        bucket["trades"] += 1
        if c.get("pnl", 0) > 0:
            bucket["wins"] += 1
        bucket["pnl"] += c.get("pnl", 0)

    out = []
    for region in _WATCH_REGIONS:
        s = stats.get(region, {"trades": 0, "wins": 0, "pnl": 0.0})
        sched = _REGION_SCHEDULES.get(region)
        out.append({
            "region": region,
            "symbol_count": len(get_symbols(region)),
            "open_positions": open_by_region.get(region, 0),
            "closed_trades": s["trades"],
            "win_rate": (s["wins"] / s["trades"] * 100) if s["trades"] else None,
            "total_pnl": round(s["pnl"], 2),
            "next_scan": _next_occurrence(*sched["scan"], sched["tz"]) if sched else None,
            "next_execute": _next_occurrence(*sched["execute"], sched["tz"]) if sched else None,
        })
    return {"regions": out}


@app.get("/api/trades.csv")
async def trades_csv():
    records = trade_log.read_all()
    closes = [r for r in records if r.get("event") == "close"]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "timestamp", "position_id", "symbol", "direction",
        "amount_usd", "entry_rate", "close_rate", "pnl",
        "duration_hours", "reason",
    ])
    for t in reversed(closes):
        writer.writerow([
            t.get("timestamp", ""),
            t.get("position_id", ""),
            t.get("symbol", ""),
            "BUY" if t.get("is_buy") else "SELL",
            t.get("amount_usd", 0),
            t.get("entry_rate", 0),
            t.get("close_rate", 0),
            t.get("pnl", 0),
            t.get("duration_hours", 0),
            t.get("reason", ""),
        ])

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=trades.csv"},
    )


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
