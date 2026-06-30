# etoroAgent

Autonomous swing-trading bot for eToro — Python 3.12, OpenAI-compatible LLM, MCP tools, APScheduler.

**Trading horizon:** 5–20 days (daily candles).  
**Universe:** ~150–200 instruments across US stocks, EU stocks, Asian ADRs, and major crypto.

---

## Architecture

```
main.py
└── Orchestrator (APScheduler)
    │
    ├── REASONING PLANE (LLM)
    │   ├── ScreeningAgent           two-speed funnel (see below)
    │   └── ResearchAgent            full ReAct loop per shortlisted symbol
    │       ├── calls MCP tools (read-only)
    │       └── produces TradingThesis (typed dataclass)
    │
    ├── EXECUTION PLANE (deterministic, no LLM)
    │   ├── risk_gate.validate()     8 hard rules — blocks trade if any fails
    │   ├── size_position()          ATR-based sizing, % of balance
    │   └── ExecutionAgent           single HTTP call to eToro, state.save()
    │
    ├── REVIEW PLANE (LLM, 1×/day per position)
    │   └── PositionReviewAgent      ReAct loop, verdict: EXIT / HOLD / TIGHTEN_STOP
    │       └── hard exit at 20 days (no LLM — deterministic)
    │
    └── MAINTENANCE (deterministic)
        ├── TrailingStopAgent        every 60 min — tighten stop, push to broker
        └── NotificationAgent        Telegram (fire-and-forget)
```

### Two-speed screening funnel

```
Full universe (~150-200 symbols)
    │
    ▼  Stage 1a — deterministic (pandas-free, pure Python)
       RSI(14) < 35 or > 65   OR   EMA20/50 cross (last 3 days)   OR   RelVol > 1.5×
    │
    ▼  Stage 1b — fast LLM (single call per batch of 8, no tools)
       Model picks top 3 per batch
    │
    ▼  Shortlist  ≤ 15 symbols
    │
    ▼  ResearchAgent — full ReAct per symbol (MCP tools, up to 8 iterations)
    │
    ▼  TradingThesis → risk_gate → ExecutionAgent
```

### MCP tool servers (read-only, stdio)

| Server | Tools provided |
|---|---|
| `etoro_server.py` | Candles, rates, instrument search |
| `indicators_server.py` | RSI, EMA, MACD, ATR, Bollinger (via `src/tools/technical.py`) |
| `finnhub_server.py` | Company news, earnings calendar |
| `cryptopanic_server.py` | Crypto news sentiment |
| `reddit_server.py` | WSB/stocks/crypto subreddit sentiment |

### State persistence

On every position open/close, `ProjectState.save()` writes `state.json` to disk.  
On startup, `ProjectState.load()` restores state and then `_reconcile_open_positions()` calls `get_portfolio()` to sync with the live broker:

- Positions closed externally (mobile app, web UI) are removed from state.
- Positions opened externally (before last restart) are added.
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
# Edit .env with your credentials
```

### 3. LLM configuration

The bot uses the **OpenAI Python SDK** (`openai` package) pointing at any OpenAI-compatible endpoint.  
There is **no LiteLLM** dependency — do not use LiteLLM model prefixes.

```bash
# LM Studio (local, recommended)
LLM_MODEL=deepseek-coder-v2-lite-instruct   # exact model ID from /v1/models
LLM_BASE_URL=http://192.168.100.216:1234/v1
LLM_API_KEY=lm-studio                        # any non-empty string

# Ollama
LLM_MODEL=qwen2.5-coder:7b
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama

# OpenAI
LLM_MODEL=gpt-4o
LLM_BASE_URL=                                # leave empty (SDK default)
LLM_API_KEY=sk-...

# Anthropic is NOT supported without adding LiteLLM as a dependency.
```

> **LM Studio tip:** The `LLM_MODEL` value must match exactly the `"id"` field from `GET /v1/models`.

### 4. Universe discovery (recommended before first run)

The static symbol lists in `src/config/universe.py` are best-guess tickers.  
eToro's internal instrument names can differ. Run discovery once to validate:

```bash
python src/config/discovery.py --regions US,EU,ASIA,CRYPTO
```

This saves `universe_cache.json` (valid 7 days). Re-run with `--force` to refresh.

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
| `WATCH_REGIONS` | `US,EU,ASIA,CRYPTO` | Regions to scan |
| `LLM_MODEL` | `deepseek-coder-v2-lite-instruct` | Model ID for research + review |
| `LLM_BASE_URL` | `http://localhost:1234/v1` | OpenAI-compatible endpoint |
| `LLM_API_KEY` | `lm-studio` | API key (any string for local) |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |
| `LLM_MAX_ITERATIONS` | `8` | Max ReAct iterations per symbol |
| `SCREENING_LLM_MODEL` | `LLM_MODEL` | Model for Stage 1b fast filter |
| `DAILY_LOSS_LIMIT_PCT` | `3.0` | Block trading beyond this % daily loss |
| `MAX_POSITION_SIZE_PCT` | `2.0` | % of balance per position |
| `MAX_OPEN_POSITIONS` | `3` | Max simultaneous positions |
| `MIN_SIGNAL_CONFIDENCE` | `0.65` | Minimum thesis confidence |
| `MIN_SIGNALS_REQUIRED` | `2` | Minimum MCP tools cited |
| `SWING_MIN_HORIZON_DAYS` | `5` | Minimum `horizon_days` in thesis |
| `SWING_MAX_HORIZON_DAYS` | `20` | Maximum `horizon_days` in thesis |
| `SWING_HARD_EXIT_DAYS` | `20` | Force-close position after N days (no LLM) |
| `SCREEN_RSI_OVERSOLD` | `35` | Stage 1a RSI oversold threshold |
| `SCREEN_RSI_OVERBOUGHT` | `65` | Stage 1a RSI overbought threshold |
| `SCREEN_REL_VOL` | `1.5` | Stage 1a relative volume multiplier |
| `SCREEN_EMA_CROSS_DAYS` | `3` | Stage 1a look-back for EMA cross |
| `REVIEW_MIN_CONFIDENCE` | `0.55` | Min confidence for position review verdict |
| `TELEGRAM_TOKEN` | | BotFather token |
| `TELEGRAM_CHAT_ID` | | Target chat/group ID |
| `FINNHUB_API_KEY` | | finnhub.io free-tier key |
| `CRYPTOPANIC_API_KEY` | | cryptopanic.com free-tier key |
| `REDDIT_CLIENT_ID` | | Reddit app client ID |
| `REDDIT_CLIENT_SECRET` | | Reddit app secret |
| `REDDIT_USER_AGENT` | `etoroAgent/1.0` | Reddit OAuth user agent |

---

## Schedule

| Job | When |
|---|---|
| US screening | 09:35 America/New_York (market open +5 min) |
| EU screening | 09:05 Europe/Berlin |
| ASIA screening | 09:05 Asia/Tokyo |
| Crypto screening | Every 6 hours UTC |
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

No network calls in tests. MCP servers and LLM are mocked.

---

## Project structure

```
EtoroAgent/
├── src/
│   ├── agents/
│   │   ├── screening_agent.py      Stage 1a (technical) + 1b (fast LLM)
│   │   ├── research_agent.py       Full ReAct loop → TradingThesis
│   │   ├── risk_gate.py            8 deterministic rules, no LLM
│   │   ├── execution_agent.py      Single-shot order submission
│   │   ├── position_review_agent.py Daily review + hard 20-day exit
│   │   ├── trailing_stop_agent.py  Tighten stop + push to broker
│   │   └── notification_agent.py   Telegram alerts
│   ├── core/
│   │   ├── etoro_client.py         Async HTTP + rate limiter + idempotent writes
│   │   ├── state.py                ProjectState + Position (save/load JSON)
│   │   ├── orchestrator.py         APScheduler wiring + startup reconciliation
│   │   ├── thesis.py               TradingThesis dataclass (LLM ↔ execution contract)
│   │   └── market_calendar.py      Trading day check (fail-closed without mcal)
│   ├── config/
│   │   ├── universe.py             Static symbol lists + cache loader
│   │   └── discovery.py            CLI to validate eToro instrument names
│   ├── llm/
│   │   └── react_runtime.py        ReAct loop driver (OpenAI SDK + DeepSeek token fallback)
│   ├── mcp_clients/
│   │   └── mcp_manager.py          Start/stop MCP servers, per-session asyncio.Lock
│   ├── mcp_servers/
│   │   ├── etoro_server.py
│   │   ├── indicators_server.py
│   │   ├── finnhub_server.py
│   │   ├── cryptopanic_server.py
│   │   └── reddit_server.py
│   └── tools/
│       └── technical.py            Pure-Python RSI, EMA, MACD, ATR, BollingerBands
├── skills/
│   └── swing_trading.md            System prompt context for the ReAct loop
├── tests/
│   ├── test_etoro_client.py
│   ├── test_market_calendar.py
│   ├── test_position_review.py
│   ├── test_react_runtime.py
│   ├── test_risk_gate.py
│   ├── test_screening_agent.py
│   └── test_technical.py
├── logs/                           Auto-created at startup
├── state.json                      Persisted position state (auto-managed)
├── universe_cache.json             Instrument ID cache (auto-managed)
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── main.py
```

---

## ⚠️ Antes de operar en real (verificación manual obligatoria)

Antes de cambiar a `ETORO_MODE=real`, validá los siguientes puntos **manualmente** en modo `demo` comparando contra la UI de eToro. El código tiene NOTEs en los lugares críticos.

### 1. Campo `stopLoss` en `open_position()`

`EtoroClient.open_position()` envía:
```python
"stopLoss": round(stop_loss_pct, 4)   # ← ¿% o precio absoluto o monto $?
```

**Qué verificar:** Abrí una posición demo con stop `stop_loss_pct = 2.0`.  
Chequeá en la UI de eToro qué stop quedó asignado.  
- Si la UI muestra `2.0` como porcentaje → el campo es correcto.  
- Si la UI muestra un precio absoluto → tenés que convertir: `stop_price = entry_rate * (1 - stop_pct/100)`.  
- Si la UI muestra un monto en dólares → calcular `stop_amount = amount_usd * stop_pct / 100`.

### 2. Endpoint y payload de `update_stop_loss()`

`TrailingStopAgent` llama a `EtoroClient.update_stop_loss()` que hace:
```python
PUT /{mode}/positions/{position_id}
{"instrumentId": ..., "stopLoss": round(new_stop, 4)}
```

**Qué verificar:**
- Que el endpoint `PUT /demo/positions/{id}` exista y acepte ese payload.
- Que después de la llamada el stop en la UI refleje el nuevo valor.
- Si el endpoint es diferente (ej. `PATCH`, o una URL distinta), actualizá `update_stop_loss()`.

### 3. Campos del portfolio en `_reconcile_open_positions()`

La reconciliación mapea campos del JSON de `GET /{mode}/portfolio`. Los nombres asumidos son:

| Campo asumido | Alternativas comunes |
|---|---|
| `positionId` | `id` |
| `instrumentName` | `ticker`, `symbol` |
| `openRate` | `openPrice`, `rate` |
| `openDateTime` | `openDate` |
| `isBuy` | `direction` (1=buy/-1=sell) |
| `amount` | `investmentAmount` |
| `currentRate` | `rate` |
| `stopLoss` | `stopLossRate` |

**Qué verificar:** Logueá la respuesta raw de `get_portfolio()` en demo y confirmá que los nombres de campo coincidan. Si no coinciden, ajustá el mapping en `_reconcile_open_positions()`.

### 4. Modo de prueba recomendado

```bash
# 1. Correr en demo durante al menos 1 ciclo completo
ETORO_MODE=demo python main.py

# 2. En los logs buscar:
#    "Reconciliation complete"  → la reconciliación funcionó
#    "position opened id=..."   → se abrió y guardó correctamente
#    "Trailing stop: ..."       → el stop se ajustó y empujó al broker

# 3. Verificar en la UI de eToro que los valores coincidan

# 4. Solo entonces cambiar a ETORO_MODE=real
```
