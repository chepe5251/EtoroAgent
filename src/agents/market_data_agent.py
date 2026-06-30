import logging
from typing import TYPE_CHECKING

from src.core.etoro_client import EtoroClient
from src.core.state import ProjectState
from src.tools.technical import compute_all

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MarketDataAgent:
    """Fetches OHLCV candles and computes technical indicators for all watched symbols."""

    def __init__(
        self,
        client: EtoroClient,
        state: ProjectState,
        symbols: list[str],
        interval: str = "M15",
        candle_count: int = 100,
    ):
        self.client = client
        self.state = state
        self.symbols = symbols
        self.interval = interval
        self.candle_count = candle_count

    async def run(self):
        logger.info("MarketDataAgent: fetching data for %s", self.symbols)
        for symbol in self.symbols:
            try:
                candles = await self.client.get_candles(
                    symbol, self.interval, self.candle_count
                )
                if not candles:
                    logger.warning("No candles returned for %s", symbol)
                    continue

                # Normalise candle keys (eToro may use different casing)
                normalised = []
                for c in candles:
                    normalised.append(
                        {
                            "open": float(c.get("open", c.get("Open", 0))),
                            "high": float(c.get("high", c.get("High", 0))),
                            "low": float(c.get("low", c.get("Low", 0))),
                            "close": float(c.get("close", c.get("Close", 0))),
                            "volume": float(c.get("volume", c.get("Volume", 0))),
                            "time": c.get("time", c.get("Time", c.get("timestamp", ""))),
                        }
                    )

                indicators = compute_all(normalised)
                self.state.market_data[symbol] = {
                    "candles": normalised,
                    "indicators": indicators,
                }
                logger.info(
                    "MarketDataAgent: %s — RSI=%.1f close=%.4f",
                    symbol,
                    indicators.get("rsi_14") or 0,
                    indicators.get("last_close") or 0,
                )
            except Exception as exc:
                logger.error("MarketDataAgent error for %s: %s", symbol, exc, exc_info=True)

        logger.info("MarketDataAgent: done — %d symbols updated", len(self.state.market_data))
