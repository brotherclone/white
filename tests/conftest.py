import pytest
import sys
import os
import shutil

from dotenv import load_dotenv
from pathlib import Path
from tests.mocks.mock_helper import redirect_test_mocks_open

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture(autouse=True)
def patch_chat_anthropic(monkeypatch):
    class FakeChatAnthropic:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setattr("app.agents.black_agent.ChatAnthropic", FakeChatAnthropic)
    redirect_test_mocks_open(monkeypatch)
    yield

def pytest_sessionstart(session):
    cache_dir = os.path.join(str(session.config.rootpath), ".pytest_cache")
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)