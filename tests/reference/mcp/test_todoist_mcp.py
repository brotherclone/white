from unittest.mock import MagicMock, patch

import pytest


def test_todoist_mcp():
    """Test Todoist MCP server imports and basic structure"""
    from app.reference.mcp.todoist.main import mcp

    # Verify the MCP server exists and has expected name
    assert mcp is not None
    assert mcp.name == "earthly_frames_todoist"


def test_get_api_client():
    """Test getting API client singleton"""
    from app.reference.mcp.todoist.main import get_api_client

    with patch.dict("os.environ", {"TODOIST_API_TOKEN": "test_token"}):
        with patch("app.reference.mcp.todoist.main.TodoistAPI") as mock_api:
            mock_instance = MagicMock()
            mock_api.return_value = mock_instance

            client = get_api_client()
            assert client is not None


def test_get_api_client_no_token():
    """Test that missing API token raises ValueError"""
    # Reset the singleton
    import app.reference.mcp.todoist.main as todoist_main
    from app.reference.mcp.todoist.main import get_api_client

    todoist_main._api_client = None

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="TODOIST_API_TOKEN not found"):
            get_api_client()


def test_get_earthly_frames_project_sections():
    """Test getting project sections"""
    from app.reference.mcp.todoist.main import get_earthly_frames_project_sections

    mock_section = MagicMock()
    mock_section.id = "section-1"
    mock_section.name = "Test Section"
    mock_section.project_id = "project-1"
    mock_section.order = 1

    with patch("app.reference.mcp.todoist.main.get_api_client") as mock_get_client:
        mock_api = MagicMock()
        mock_api.get_sections.return_value = [mock_section]
        mock_get_client.return_value = mock_api

        result = get_earthly_frames_project_sections()

        assert isinstance(result, list)
        # The function should process sections and return dict format
