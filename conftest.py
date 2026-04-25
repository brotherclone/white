import os

import pytest


@pytest.fixture(autouse=True)
def set_mock_mode(monkeypatch):
    if "MOCK_MODE" not in os.environ:
        monkeypatch.setenv("MOCK_MODE", "true")
    yield
