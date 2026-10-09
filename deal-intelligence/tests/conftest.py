"""Shared fixtures. Everything runs offline: Hindsight and the model are replaced by fakes
(tests/fakes.py for memory, tests/fake_llm.py for the model)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from deal_intelligence.config import Settings


@pytest.fixture(autouse=True)
def _isolated_workspaces(tmp_path, monkeypatch):
    """Tests never read or write the real workspace registry (data/workspaces.json)."""
    monkeypatch.setenv("DEAL_WORKSPACES_FILE", str(tmp_path / "workspaces.json"))


@pytest.fixture
def settings(tmp_path) -> Settings:
    return replace(Settings(), db_path=tmp_path / "deals.db", uploads_dir=tmp_path / "uploads")
