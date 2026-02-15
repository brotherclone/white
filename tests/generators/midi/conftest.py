"""Local conftest for MIDI generator tests.

Overrides the root conftest's autouse fixture that requires langchain_anthropic,
which is not available in .venv312.
"""

import pytest


@pytest.fixture(autouse=True)
def patch_chat_anthropic():
    """No-op override — MIDI tests don't need LangChain mocking."""
    yield
