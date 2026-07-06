"""
Global test isolation: no test should ever touch the real state.json or
trades.jsonl in the project root. Both modules resolve their file path
relative to the process cwd, which is the repo root when pytest runs here —
without this fixture, any test that exercises ExecutionAgent/ProjectState
persistence silently pollutes the live bot's files.
"""
import pytest

from src.core import state as state_module
from src.core import trade_log as trade_log_module


@pytest.fixture(autouse=True)
def _isolate_persistence_files(tmp_path, monkeypatch):
    monkeypatch.setattr(state_module, "_STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(trade_log_module, "_TRADE_LOG_FILE", tmp_path / "trades.jsonl")
