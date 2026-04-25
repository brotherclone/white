import pytest


@pytest.fixture(autouse=True)
def set_mock_mode(monkeypatch):
    monkeypatch.setenv("MOCK_MODE", "true")
    yield
