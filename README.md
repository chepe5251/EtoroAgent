# etoroAgent

Autonomous multi-agent trading system for eToro built with Python 3.11+, LiteLLM, and APScheduler.

## Architecture

```
main.py
└── Orchestrator (APScheduler loop)
    ├── MarketDataAgent   → fetches candles + computes RSI/MACD/EMA/BB/ATR
    ├── SentimentAgent    → NewsAPI + Reddit → LLM sentiment score
    ├── SignalAgent       → LLM decides BUY/SELL/HOLD per symbol
    ├── DecisionAgent     → position sizing, max-positions, dedup
    ├── RiskAgent         → daily loss limit, trailing stop adjustment
    ├── ExecutionAgent    → calls eToro API to open/close positions
    └── NotificationAgent → Telegram alerts
```

The LLM is used only in **SignalAgent** (trading signals) and **SentimentAgent** (news/Reddit scoring). All other agents are deterministic.

## Setup

### 1. Clone and create environment

```bash
git clone <repo>
cd etoroBot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your credentials
```

| Variable | Description |
|---|---|
| `ETORO_MODE` | `demo` or `real` |
| `ETORO_PUBLIC_API_KEY` | eToro API key (`x-api-key` header) |
| `ETORO_USER_KEY` | eToro user key JWT (`x-user-key` header) |
| `WATCH_SYMBOLS` | Comma-separated tickers: `BTC,ETH,AAPL,TSLA` |
| `LLM_MODEL` | LiteLLM model string (see below) |
| `LLM_BASE_URL` | Override for local models (LM Studio / ollama) |
| `LLM_API_KEY` | API key for the LLM provider |
| `DAILY_LOSS_LIMIT_PCT` | Block trading if daily loss exceeds this % of balance (default `3.0`) |
| `MAX_POSITION_SIZE_PCT` | % of balance per trade (default `2.0`) |
| `MAX_OPEN_POSITIONS` | Max simultaneous open positions (default `3`) |
| `TELEGRAM_TOKEN` | BotFather token for notifications |
| `TELEGRAM_CHAT_ID` | Target chat/group ID |
| `NEWS_API_KEY` | newsapi.org key |
| `REDDIT_CLIENT_ID` | Reddit app client ID |
| `REDDIT_CLIENT_SECRET` | Reddit app secret |

### 3. LLM configuration examples

```bash
# LM Studio (PC local con RTX 5060 Ti)  ← recomendado para uso local
LLM_MODEL=lm_studio/qwen3-8b      # prefijo lm_studio/ + nombre del modelo en LM Studio
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio              # cualquier string, LM Studio no valida la key

# Ollama (local)
LLM_MODEL=ollama/qwen3:8b
LLM_BASE_URL=http://localhost:11434
LLM_API_KEY=ollama

# OpenAI
LLM_MODEL=gpt-4o
LLM_BASE_URL=                      # dejar vacío
LLM_API_KEY=sk-...

# Anthropic
LLM_MODEL=claude-opus-4-8
LLM_BASE_URL=                      # dejar vacío
LLM_API_KEY=sk-ant-...
```

> **Tip LM Studio:** el `LLM_MODEL` debe tener el prefijo `lm_studio/` seguido exactamente del identificador que LM Studio muestra en "Model Name" (p.ej. `lm_studio/qwen3-8b-instruct`). Si el modelo cambia, solo actualiza esta variable.

## Running locally

```bash
# Demo mode (safe — no real money)
ETORO_MODE=demo python main.py

# Real mode
ETORO_MODE=real python main.py
```

Logs are written to `logs/etoroAgent.log` and stdout.

## Switching demo → real

Edit your `.env` file:
```
ETORO_MODE=real
```

Then restart the agent. The client automatically routes execution endpoints to `/real/positions` instead of `/demo/positions`.

## Running with Docker (VPS / Contabo)

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f etoro-agent

# Stop
docker-compose down
```

Logs are persisted to `./logs/` on the host.

## Running tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

## Cycle schedule

| Job | Interval |
|---|---|
| Main cycle (data → signal → execute) | Every 15 min (configurable) |
| Trailing stop adjustment | Every 5 min (configurable) |
| Daily P&L summary (Telegram) | 23:00 UTC |

## Risk controls

- **Daily loss limit**: if cumulative P&L drops below `-DAILY_LOSS_LIMIT_PCT %` of balance, all new trades are blocked until midnight UTC.
- **Max open positions**: capped at `MAX_OPEN_POSITIONS` (default 3).
- **No duplicate positions**: only one position per symbol at a time.
- **Stop-loss**: initial stop = 1.5 × ATR(14); tightened automatically as price moves in your favour (trailing every 5 min).
- **Confidence filter**: signals with confidence < 65% are discarded before reaching DecisionAgent.

## Project structure

```
etoroBot/
├── src/
│   ├── agents/
│   │   ├── market_data_agent.py   # OHLCV fetch + indicators
│   │   ├── sentiment_agent.py     # news + Reddit → LLM score
│   │   ├── signal_agent.py        # LLM → BUY/SELL/HOLD
│   │   ├── decision_agent.py      # sizing + pre-validation
│   │   ├── execution_agent.py     # eToro API calls
│   │   ├── risk_agent.py          # daily loss + trailing stops
│   │   └── notification_agent.py  # Telegram
│   ├── core/
│   │   ├── etoro_client.py        # async HTTP + rate limiting + retry
│   │   ├── state.py               # shared ProjectState dataclasses
│   │   └── orchestrator.py        # APScheduler wiring
│   └── tools/
│       ├── technical.py           # pure indicator functions
│       └── sentiment.py           # news/reddit fetch helpers
├── tests/
│   ├── test_etoro_client.py
│   ├── test_risk_agent.py
│   └── test_technical.py
├── logs/                          # auto-created
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── main.py
```
