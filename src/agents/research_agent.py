"""
ResearchAgent — the ONLY agentic component in the system.
Runs a free ReAct loop with read-only tools to investigate a symbol
and produce a structured TradingThesis.

The LLM has zero authority to move money. It can only call read-only
tools (candles, indicators, news, sentiment) and must justify its thesis
using at least 2 independent data sources.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.thesis import TradingThesis
from src.llm.react_runtime import ReActRuntime

if TYPE_CHECKING:
    from src.mcp_clients.mcp_manager import MCPManager

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).parent.parent.parent / "skills"

_SYSTEM_PROMPT_CORE = """\
Sos un analista de SWING TRADING (horizonte 5-20 días). Tenés acceso a
herramientas de datos de mercado, indicadores técnicos y sentiment.
Tu objetivo es detectar movimientos sostenidos de varios días, no scalping.

REGLAS ESTRICTAS (no negociables):
1. NO podés abrir ni cerrar posiciones. Solo investigás y concluís.
2. Usá candles D1 (diarias) para el análisis principal — el contexto es swing, no intraday.
3. Tu conclusión DEBE estar respaldada por AL MENOS 2 fuentes de datos
   distintas que sean COHERENTES entre sí.
   Ejemplos válidos:
     - RSI sobrevendido en D1 + sentiment positivo en noticias
     - MACD cruce alcista en D1 + EMA20 cruzando EMA50 al alza
     - RSI extremo + volumen > 1.5x promedio (momentum real)
   Una sola señal NUNCA es suficiente para un trade swing.
4. Si las señales son contradictorias, conclusión = "hold".
   No operar es una decisión válida y a veces la correcta.
5. Incluí una condición de invalidación clara: ¿qué cambio de mercado
   invalidaría esta tesis? (ej: "Si el precio cierra por debajo de EMA50",
   "Si RSI vuelve por encima de 50 en próximas 48h").
6. Estimá el horizonte realista en días (entre 5 y 20).
7. Tu respuesta final DEBE ser un JSON válido con este schema exacto:
{
  "symbol": "<TICKER>",
  "action": "buy" | "sell" | "hold",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<explicación citando qué tools usaste y qué encontraste>",
  "signals_used": ["<tool_name_1>", "<tool_name_2>", ...],
  "suggested_stop_loss_atr_multiple": <float, default 1.5>,
  "horizon_days": <int entre 5 y 20>,
  "invalidation_condition": "<condición concreta que invalidaría esta tesis>"
}

NO incluyas nada fuera del JSON en tu respuesta final.
"""


class ResearchAgent:
    """
    Investigates a symbol using a ReAct loop with read-only MCP tools.
    Produces a TradingThesis that the deterministic risk gate then validates.
    """

    def __init__(self, mcp_manager: "MCPManager"):
        self.mcp_manager = mcp_manager
        self._skills_text: str | None = None

    def _load_skills(self) -> str:
        """Load and cache skills markdown files as context."""
        if self._skills_text is not None:
            return self._skills_text
        parts: list[str] = []
        if _SKILLS_DIR.exists():
            for md_file in sorted(_SKILLS_DIR.glob("*.md")):
                parts.append(f"## {md_file.stem.replace('_', ' ').title()}\n\n{md_file.read_text()}")
        self._skills_text = "\n\n---\n\n".join(parts) if parts else ""
        return self._skills_text

    async def run(self, symbol: str) -> TradingThesis | None:
        """
        Run the ReAct investigation loop for a single symbol.

        Returns a TradingThesis or None if the LLM failed to produce
        a parseable result.
        """
        symbol = symbol.upper()
        logger.info("ResearchAgent: starting investigation for %s", symbol)

        skills = self._load_skills()
        system_prompt = _SYSTEM_PROMPT_CORE
        if skills:
            system_prompt += f"\n\n---\n\n# Guías de análisis\n\n{skills}"

        runtime = ReActRuntime(
            mcp_manager=self.mcp_manager,
            system_prompt=system_prompt,
        )

        user_prompt = (
            f"Investigá {symbol} para SWING TRADING (horizonte 5-20 días). "
            f"Empezá con candles D1 para tendencia principal. "
            f"Chequeá RSI, EMA20/EMA50, MACD en timeframe diario. "
            f"Luego validá con sentiment (noticias, redes). "
            f"Si encontrás señal, estimá horizonte y condición de invalidación. "
            f"Concluí con el JSON de la tesis."
        )

        result = await runtime.run(user_prompt)

        if result.get("error"):
            logger.warning(
                "ResearchAgent: ReAct error for %s: %s", symbol, result["error"]
            )

        thesis_dict = result.get("thesis")
        if not thesis_dict:
            logger.error("ResearchAgent: no parseable thesis for %s", symbol)
            return None

        try:
            thesis = TradingThesis.from_dict(thesis_dict)
            logger.info("ResearchAgent: thesis for %s → %s", symbol, thesis)
            return thesis
        except Exception as exc:
            logger.error(
                "ResearchAgent: thesis parse error for %s: %s (raw=%s)",
                symbol, exc, thesis_dict,
            )
            return None
