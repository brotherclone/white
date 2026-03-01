import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

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


# --- venv312 re-run support ---


def _find_venv312_python() -> str | None:
    """Return the absolute path to .venv312/bin/python if present, else None."""
    root = Path(__file__).resolve().parent.parent
    candidate = root / ".venv312" / "bin" / "python"
    if candidate.exists() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def pytest_collection_modifyitems(config, items):
    """If any collected tests are marked with `venv312`, re-run them under
    `.venv312/bin/python` (if present) and then deselect them from the main
    run. This allows tests that require a different venv to pass without
    forcing the entire test suite to run under the alternate interpreter.
    """
    venv_items = [it for it in items if it.get_closest_marker("venv312")]
    if not venv_items:
        return

    current_py = sys.executable
    venv_py = _find_venv312_python()
    if venv_py is None:
        # No alternate venv available — let tests run in current interpreter.
        return

    # Detect if we're already running under .venv312 through VIRTUAL_ENV or path.
    virtual_env = os.environ.get("VIRTUAL_ENV")
    current_in_venv312 = False
    if virtual_env and os.path.abspath(virtual_env).endswith(".venv312"):
        current_in_venv312 = True
    elif ".venv312" in os.path.abspath(current_py):
        current_in_venv312 = True

    if current_in_venv312:
        return

    nodeids = [it.nodeid for it in venv_items]
    cmd = [venv_py, "-m", "pytest", "-q", "--maxfail=1"] + nodeids
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    # Deselect those items so the main run doesn't execute them again
    deselected = set(nodeids)
    remaining = [it for it in items if it.nodeid not in deselected]
    items[:] = remaining


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "venv312: mark test to run under .venv312 python"
    )
