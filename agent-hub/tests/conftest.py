"""Shared fixtures: an isolated data folder, fake Deal and RFP apps, and an Engine wired to them."""

from __future__ import annotations

import pytest

from agent_hub import store as store_module
from agent_hub.clients import DealClient, RfpClient
from agent_hub.engine import Engine
from agent_hub.router import load_registry
from agent_hub.store import Store

from .fakes import Fakes

AGENTS = load_registry()


@pytest.fixture(autouse=True)
def _isolated_data(tmp_path, monkeypatch):
    """No test may touch the real data/hub.db or data/uploads. Rate limits are off unless a test turns them on."""
    monkeypatch.setattr(store_module, "DATA_DIR", tmp_path / "default-data")
    monkeypatch.setenv("HUB_RATE_LIMIT", "off")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def fakes() -> Fakes:
    return Fakes()


def make_engine(fakes: Fakes, data_dir, **kwargs) -> Engine:
    by_id = {a["id"]: a for a in AGENTS}
    return Engine(
        Store(data_dir), DealClient(by_id["deals"], transport=fakes.deal.transport()),
        RfpClient(by_id["rfp"], transport=fakes.rfp.transport()), AGENTS,
        **{"poll_interval": 0.01, "max_down_polls": 5, **kwargs},
    )


@pytest.fixture
def engine(fakes, tmp_path) -> Engine:
    eng = make_engine(fakes, tmp_path / "hub-data")
    yield eng
    eng.store.close()
