import pytest


@pytest.fixture(autouse=True)
def set_mock_mode(monkeypatch):
    # Always override — load_dotenv() in agent modules sets MOCK_MODE=False
    # from .env before tests run, so the os.environ check would be bypassed.
    monkeypatch.setenv("MOCK_MODE", "true")
    yield
