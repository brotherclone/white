from unittest.mock import MagicMock, patch


def test_discogs_mcp():
    """Test Discogs MCP server imports and basic structure"""
    from app.reference.mcp.discogs.main import mcp

    # Verify the MCP server exists and has expected name
    assert mcp is not None
    assert mcp.name == "earthly_frames_discogs"


async def test_discogs_look_up_artist_by_name():
    """Test looking up artist by name"""
    from app.reference.mcp.discogs.main import look_up_artist_by_name

    # Mock the discogs client
    with patch(
        "app.reference.mcp.discogs.main.discogs_earthly_frames_retriever_service"
    ) as mock_service:
        mock_discogs = MagicMock()
        mock_artist = MagicMock()
        mock_artist.id = 12345
        mock_artist.name = "Test Artist"
        mock_artist.type = "artist"
        mock_artist.url = "http://example.com"

        mock_discogs.search.return_value = [mock_artist]
        mock_service.return_value = mock_discogs

        result = await look_up_artist_by_name("Test Artist")

        assert result is not None
        assert result["name"] == "Test Artist"
        assert result["id"] == 12345


async def test_discogs_look_up_artist_no_results():
    """Test looking up artist with no results"""
    from app.reference.mcp.discogs.main import look_up_artist_by_name

    with patch(
        "app.reference.mcp.discogs.main.discogs_earthly_frames_retriever_service"
    ) as mock_service:
        mock_discogs = MagicMock()
        mock_discogs.search.return_value = []
        mock_service.return_value = mock_discogs

        result = await look_up_artist_by_name("Nonexistent Artist")

        assert result is None
