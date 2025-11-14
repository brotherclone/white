from unittest.mock import AsyncMock, patch


def test_earthly_frames_mcp():
    """Test Earthly Frames MCP server imports and basic structure"""
    from app.reference.mcp.earthly_frames.main import mcp

    # Verify the MCP server exists and has expected name
    assert mcp is not None
    assert mcp.name == "earthly_frames"


async def test_get_all_recordings():
    """Test getting all recordings"""
    from app.reference.mcp.earthly_frames.main import get_all_recordings

    mock_data = [
        {"id": 1, "title": "Album 1", "rainbow_table": "red"},
        {"id": 2, "title": "Album 2", "rainbow_table": "not_associated"},
        {"id": 3, "title": "Album 3", "rainbow_table": "blue"},
    ]

    with patch(
        "app.reference.mcp.earthly_frames.main.earthly_frames_retriever_service",
        new_callable=AsyncMock,
    ) as mock_service:
        mock_service.return_value = mock_data

        # Test without filter
        result = await get_all_recordings(filter_for_rainbow_table=False)
        assert len(result) == 3

        # Test with filter
        result_filtered = await get_all_recordings(filter_for_rainbow_table=True)
        assert len(result_filtered) == 2
        assert all(r["rainbow_table"] != "not_associated" for r in result_filtered)


async def test_album_for_color():
    """Test getting album for specific color"""
    from app.reference.mcp.earthly_frames.main import album_for_color

    mock_data = [
        {"id": 1, "title": "Album 1", "rainbow_table": "red"},
        {"id": 2, "title": "Album 2", "rainbow_table": "blue"},
        {"id": 3, "title": "Album 3", "rainbow_table": "red"},
    ]

    with patch(
        "app.reference.mcp.earthly_frames.main.earthly_frames_retriever_service",
        new_callable=AsyncMock,
    ) as mock_service:
        mock_service.return_value = mock_data

        result = await album_for_color("red")
        assert len(result) == 2
        assert all(a["rainbow_table"] == "red" for a in result)


async def test_get_all_recordings_error():
    """Test error handling when service fails"""
    from app.reference.mcp.earthly_frames.main import get_all_recordings

    with patch(
        "app.reference.mcp.earthly_frames.main.earthly_frames_retriever_service",
        new_callable=AsyncMock,
    ) as mock_service:
        mock_service.return_value = None

        result = await get_all_recordings(filter_for_rainbow_table=False)
        assert result == "Unable to retrieve recordings"
