import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


def test_lucid_nonsense_access_mcp():
    """Test Lucid Nonsense Access MCP server imports and basic structure"""
    from app.reference.mcp.lucid_nonsense_access.main import mcp

    # Verify the MCP server exists and has expected name
    assert mcp is not None
    assert mcp.name == "lucid_nonsense_access"


def test_get_base_path():
    """Test getting base path from environment"""
    from app.reference.mcp.lucid_nonsense_access.main import get_base_path

    with patch.dict("os.environ", {"LUCID_NONSENSE_PATH": "/test/path"}):
        path = get_base_path()
        assert path == Path("/test/path")


def test_get_base_path_default():
    """Test getting default base path"""
    from app.reference.mcp.lucid_nonsense_access.main import get_base_path

    with patch.dict("os.environ", {}, clear=True):
        path = get_base_path()
        assert path == Path("/Volumes/LucidNonsense/White")


def test_search_white_project_files():
    """Test searching for white project files"""
    from app.reference.mcp.lucid_nonsense_access.main import search_white_project_files

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test files
        test_files = ["file1.txt", "file2.py", "file3.md"]
        for f in test_files:
            Path(tmpdir, f).touch()

        with patch(
            "app.reference.mcp.lucid_nonsense_access.main.get_base_path"
        ) as mock_path:
            mock_path.return_value = Path(tmpdir)

            result = search_white_project_files()

            assert isinstance(result, list)
            assert len(result) == 3
            assert "file1.txt" in result


def test_search_white_project_files_error():
    """Test error handling when directory doesn't exist"""
    from app.reference.mcp.lucid_nonsense_access.main import search_white_project_files

    with patch(
        "app.reference.mcp.lucid_nonsense_access.main.get_base_path"
    ) as mock_path:
        mock_path.return_value = Path("/nonexistent/path")

        result = search_white_project_files()

        assert result == []


def test_open_lucid_nonsense_file():
    """Test opening and reading a file"""
    from app.reference.mcp.lucid_nonsense_access.main import open_lucid_nonsense_file

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir, "test.txt")
        test_content = "Hello, World!"
        test_file.write_text(test_content)

        with patch(
            "app.reference.mcp.lucid_nonsense_access.main.get_base_path"
        ) as mock_path:
            mock_path.return_value = Path(tmpdir)

            result = open_lucid_nonsense_file("test.txt")

            assert result == test_content


def test_open_lucid_nonsense_file_security():
    """Test security check prevents path traversal"""
    from app.reference.mcp.lucid_nonsense_access.main import open_lucid_nonsense_file

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch(
            "app.reference.mcp.lucid_nonsense_access.main.get_base_path"
        ) as mock_path:
            mock_path.return_value = Path(tmpdir)

            with pytest.raises(ValueError, match="outside allowed directory"):
                open_lucid_nonsense_file("../../../etc/passwd")


def test_open_lucid_nonsense_file_not_found():
    """Test error when file doesn't exist"""
    from app.reference.mcp.lucid_nonsense_access.main import open_lucid_nonsense_file

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch(
            "app.reference.mcp.lucid_nonsense_access.main.get_base_path"
        ) as mock_path:
            mock_path.return_value = Path(tmpdir)

            with pytest.raises(OSError):
                open_lucid_nonsense_file("nonexistent.txt")
