# etoroAgent

Autonomous swing-trading bot for eToro — Python 3.12, APScheduler.

**100% rule-based. No LLM anywhere in the trading pipeline.** Every entry, exit,
and sizing decision is deterministic Python, backtested against 5 years of real
eToro data (real fees, real weekend carry, real leverage) before being wired
into production. See `src/backtest/` for the validation.

**Strategy:** trend-following breakout/pullback. EMA50>EMA200 trend gate, entry
on a Donchian breakout or an EMA20 pullback-resume (both volume-confirmed),
exit on trend break (close < EMA50), stop-loss, or a 20-day hard time limit.

**Validated result:** out-of-sample profit factor 1.60 across 140 real symbols,
5 years, with full transaction costs and 5x leverage (see
`src/backtest/engine.py` and `src/backtest/run_backtest.py`).

**Trading horizon:** 5–20 days (daily candles).
**Universe:** US/EU/ASIA stocks — large-caps, mid-caps, and momentum names.
Crypto is excluded: eToro's fee structure (1% spread/side + weekend-tripled
overnight carry) made it unprofitable in backtesting regardless of leverage.

---

## Architecture

```
main.py
└── Orchestrator (APScheduler)
    │
    ├── SIGNAL PLANE (deterministic, no LLM)
    │   ├── ScreeningAgent           EMA50>EMA200 + breakout/pullback + volume
    │   └── thesis_builder           builds a TradingThesis directly from the signal
    │
    ├── EXECUTION PLANE (deterministic, no LLM)
    │   ├── risk_gate.validate()     8 hard rules — blocks trade if any fails
    │   ├── size_position()          risk-based sizing (ATR stop distance), leverage-aware
    │   └── ExecutionAgent           single HTTP call to eToro, state.save()
    │
    ├── REVIEW PLANE (deterministic, 1×/day per position)
    │   └── PositionReviewAgent      exit if close < EMA50 (trend break)
    │       └── hard exit at 20 days regardless
    │
    └── MAINTENANCE (deterministic)
        ├── TrailingStopAgent        every 60 min — tighten stop, push to broker
        └── NotificationAgent        Telegram (fire-and-forget)
```

### Screening funnel

```
Full universe (~140 symbols across US/EU/ASIA)
    │
    ▼  Deterministic filter (pandas-free, pure Python)
       Trend gate: EMA50 > EMA200
       Entry: Donchian breakout (20-bar high)  OR  EMA20 pullback-resume
       Both require relative volume > 1.5× the 20-day average
    │
    ▼  Shortlist  ≤ 15 symbols, in universe order
    │
    ▼  thesis_builder.build_thesis() → TradingThesis (fixed confidence, templated
    │  reasoning citing the validated backtest, ATR-based stop, 15-day horizon)
    │
    ▼  risk_gate → ExecutionAgent
```

### Position sizing (must match the backtest — see `src/backtest/engine.py`)

```
stop_distance = suggested_stop_loss_atr_multiple × ATR
risk_amount   = balance × RISK_PER_TRADE_PCT%
notional      = (risk_amount / stop_distance) × current_price
notional      = min(notional, balance × MAX_POSITION_SIZE_PCT% × LEVERAGE)

broker margin sent to eToro = notional / LEVERAGE   (real leverage flag set on the order)
Position.amount_usd (stored) = notional             (so P&L accounting matches the backtest)
```

If you change `RISK_PER_TRADE_PCT`, `MAX_POSITION_SIZE_PCT`, or `LEVERAGE`,
re-run the backtest with matching flags before trusting the new numbers —
these aren't independent knobs, they're the exact parameters that were validated.

### MCP tool servers (read-only, stdio)

Only one server is actually used in production — the rest of the original
five (etoro/finnhub/cryptopanic/reddit/exa) supported the LLM ReAct research
loop that has since been replaced by `thesis_builder.py` and are no longer
started (see `src/mcp_clients/mcp_manager.py`).

| Server | Tools provided | Used for |
|---|---|---|
| `indicators_server.py` | RSI, EMA, MACD, ATR, Bollinger (via `src/tools/technical.py`) | `orchestrator._fetch_price_and_atr` (price/ATR for sizing) |

### State persistence

On every position open/close, `ProjectState.save()` writes `state.json` to disk.
On startup, `ProjectState.load()` restores state and then `_reconcile_open_positions()` calls `get_portfolio()` to sync with the live broker:

- Positions closed externally (mobile app, web UI) are removed from state.
- Positions opened externally are logged but NOT auto-adopted (safety: the bot
  only manages positions it opened itself).
- Current rates are refreshed from the broker response.

### Risk gate rules (all must pass)

| # | Rule |
|---|---|
| 0 | `action == "hold"` → skip (no trade needed) |
| 1 | `confidence ≥ MIN_SIGNAL_CONFIDENCE` (default 65%) |
| 2 | `len(signals_used) ≥ MIN_SIGNALS_REQUIRED` (default 2) |
| 3 | `len(reasoning) ≥ 50` chars |
| 4 | No active daily-loss block |
| 5 | `realized_loss + unrealized_loss < DAILY_LOSS_LIMIT_PCT` of balance |
| 6 | `open_positions < MAX_OPEN_POSITIONS` (default 3) |
| 7 | No duplicate symbol already open |
| 8 | `5 ≤ horizon_days ≤ 20` |

---

## Backtesting

Everything the live bot does was validated first. Three engines live under `src/backtest/`:

| Module | Strategy | Status |
|---|---|---|
| `engine.py` + `run_backtest.py` | Breakout/pullback trend-following (long-only) | **Validated — this is what's deployed.** OOS PF 1.60, 140 symbols, 5y. |
| `market_structure.py` + `run_market_structure.py` | BOS/ChoCh swing-structure (long-only) | Positive but weaker (OOS PF ~1.2). Not deployed. |
| `first_red_day.py` + `run_first_red_day.py` | Parabolic-reversal short (day/swing) | Negative on both blue-chips and volatile mid-caps. Not deployed. |

```bash
# Fetch 5y of real candles and run the validated strategy
python -m src.backtest.run_backtest --fetch --years 5 \
  --symbols AAPL MSFT NVDA ... \
  --equity 800 --leverage 5 --risk-pct 1.0 \
  --trend-filter-type ema50_200 --breakout --pullback \
  --no-rsi-signal --no-ema-signal --exit trend_break
```

Key honesty features baked into the engine (see its module docstring for detail):
mark-to-market equity curve, gap-through stop fills, per-asset-class transaction
costs, weekend-aware overnight carry (Friday nights charged 3x), and an IS/OOS
split so results can't be silently overfit.

---

## Setup

### 1. Clone and create environment

```bash
git clone git@github.com:chepe5251/EtoroAgent.git
cd EtoroAgent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your eToro credentials
```

### 3. Universe discovery (recommended before first run)

The static symbol lists in `src/config/universe.py` are best-guess tickers.
eToro's internal instrument names/IDs can differ. Run discovery once to validate:

```bash
python src/config/discovery.py --regions US,EU,ASIA
```

This saves `universe_cache.json`. Re-run with `--force` to refresh.

---

## Running

```bash
# Demo mode (safe — no real money)
ETORO_MODE=demo python main.py

# Real mode
ETORO_MODE=real python main.py
```

Logs go to `logs/etoroAgent.log` and stdout.

## Running with Docker

```bash
docker-compose up -d
docker-compose logs -f etoro-agent
docker-compose down
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ETORO_MODE` | `demo` | `demo` or `real` |
| `ETORO_PUBLIC_API_KEY` | *(required)* | `x-api-key` header |
| `ETORO_USER_KEY` | *(required)* | `x-user-key` JWT header |
| `WATCH_REGIONS` | `US,EU,ASIA` | Regions to scan (crypto excluded — see above) |
| `DAILY_LOSS_LIMIT_PCT` | `3.0` | Block trading beyond this % daily loss |
| `RISK_PER_TRADE_PCT` | `1.0` | % of balance risked per trade (at stop distance) |
| `MAX_POSITION_SIZE_PCT` | `10.0` | Notional cap as % of balance, before leverage |
| `LEVERAGE` | `1.0` | Real CFD leverage applied to the notional cap |
| `MAX_OPEN_POSITIONS` | `3` | Max simultaneous positions |
| `MIN_SIGNAL_CONFIDENCE` | `0.65` | Minimum thesis confidence |
| `MIN_SIGNALS_REQUIRED` | `2` | Minimum signals cited in thesis |
| `SWING_MIN_HORIZON_DAYS` | `5` | Minimum `horizon_days` in thesis |
| `SWING_MAX_HORIZON_DAYS` | `20` | Maximum `horizon_days` in thesis |
| `SWING_HARD_EXIT_DAYS` | `20` | Force-close position after N days |
| `SCREEN_REL_VOL` | `1.5` | Relative volume multiplier threshold |
| `SCREEN_DONCHIAN_LOOKBACK` | `20` | Breakout lookback in bars |
| `TELEGRAM_TOKEN` | | BotFather token |
| `TELEGRAM_CHAT_ID` | | Target chat/group ID |

---

## Schedule

| Job | When |
|---|---|
| US screening | 09:35 America/New_York (market open +5 min) |
| EU screening | 09:05 Europe/Berlin |
| ASIA screening | 09:05 Asia/Tokyo |
| Position review | 07:00 UTC daily |
| Trailing stop adjustment | Every 60 minutes |
| Daily P&L summary (Telegram) | 23:00 UTC |

All equity schedules are skipped on market holidays via `pandas-market-calendars`.
If the library is not installed, equity markets are treated as **CLOSED** (fail-safe).

---

## Running tests

```bash
pytest tests/ -q
```

No network calls in tests.

---

## Project structure

```
EtoroAgent/
├── src/
│   ├── agents/
│   │   ├── screening_agent.py      Deterministic technical filter (no LLM)
│   │   ├── thesis_builder.py       Builds TradingThesis directly from the signal
│   │   ├── risk_gate.py            8 deterministic rules
│   │   ├── execution_agent.py      Risk-based sizing + single-shot order submission
│   │   ├── position_review_agent.py Daily trend-break check + hard 20-day exit
│   │   ├── trailing_stop_agent.py  Tighten stop + push to broker
│   │   └── notification_agent.py   Telegram alerts
│   ├── backtest/
│   │   ├── engine.py               Validated breakout/pullback engine (deployed strategy)
│   │   ├── market_structure.py     BOS/ChoCh engine (not deployed)
│   │   ├── first_red_day.py        Short-side parabolic-reversal engine (not deployed)
│   │   ├── data.py                 Candle fetch/cache
│   │   └── metrics.py              Win rate, profit factor, drawdown, Sharpe
│   ├── core/
│   │   ├── etoro_client.py         Async HTTP + rate limiter + idempotent writes
│   │   ├── state.py                ProjectState + Position (save/load JSON)
│   │   ├── orchestrator.py         APScheduler wiring + startup reconciliation
│   │   ├── thesis.py               TradingThesis dataclass (signal ↔ execution contract)
│   │   └── market_calendar.py      Trading day check (fail-closed without mcal)
│   ├── config/
│   │   ├── universe.py             Static symbol lists + cache loader
│   │   └── discovery.py            CLI to validate eToro instrument names
│   ├── mcp_clients/
│   │   └── mcp_manager.py          Start/stop MCP servers (only indicators_server.py now)
│   ├── mcp_servers/
│   │   └── indicators_server.py    RSI/EMA/MACD/ATR/Bollinger tool
│   └── tools/
│       └── technical.py            Pure-Python RSI, EMA, MACD, ATR, BollingerBands
├── tests/
├── logs/                           Auto-created at startup
├── state.json                      Persisted position state (auto-managed)
├── universe_cache.json             Instrument ID cache (auto-managed)
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── main.py
```
