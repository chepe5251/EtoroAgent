"""
TradingThesis — the structured output produced by thesis_builder.build_thesis().
This is the contract between the agentic reasoning plane and the
deterministic execution plane. Every field must be present and typed;
the risk gate validates this schema before any order is considered.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TradingThesis:
    symbol: str
    action: str                          # "buy" | "sell" | "hold"
    confidence: float                    # 0.0 – 1.0
    reasoning: str                       # human-readable explanation
    signals_used: list[str]              # tool names that back the thesis
    suggested_stop_loss_atr_multiple: float = 1.5  # how many ATRs for stop
    horizon_days: int = 10               # expected holding period 5-20 days
    invalidation_condition: str = ""    # what would kill the thesis

    # ------------------------------------------------------------------ #
    # Constructors / serialisation
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TradingThesis":
        action = str(d.get("action", "hold")).lower()
        if action not in ("buy", "sell", "hold"):
            action = "hold"
        return cls(
            symbol=str(d["symbol"]).upper(),
            action=action,
            confidence=float(d.get("confidence", 0.0)),
            reasoning=str(d.get("reasoning", "")),
            signals_used=list(d.get("signals_used", [])),
            suggested_stop_loss_atr_multiple=float(
                d.get("suggested_stop_loss_atr_multiple", 1.5)
            ),
            horizon_days=int(d.get("horizon_days", 10)),
            invalidation_condition=str(d.get("invalidation_condition", "")),
        )

    @classmethod
    def from_json(cls, text: str) -> "TradingThesis":
        """Parse a thesis from a raw JSON string (with optional markdown fences)."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # strip opening fence (```json or ```)
            lines = lines[1:]
            # strip closing fence
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "signals_used": self.signals_used,
            "suggested_stop_loss_atr_multiple": self.suggested_stop_loss_atr_multiple,
            "horizon_days": self.horizon_days,
            "invalidation_condition": self.invalidation_condition,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    def is_actionable(self) -> bool:
        return self.action in ("buy", "sell")

    def __str__(self) -> str:
        return (
            f"TradingThesis({self.symbol} {self.action.upper()} "
            f"confidence={self.confidence:.0%} signals={self.signals_used})"
        )
