import base64
import os
import tempfile
from unittest.mock import MagicMock, patch


def test_mcp_midi_mate():
    """Test MIDI Mate MCP server imports and basic structure"""
    from app.reference.mcp.midi_mate.main import mcp

    # Verify the MCP server exists and has expected name
    assert mcp is not None
    assert mcp.name == "midi_mate"


def test_save_midi_from_base64_basic():
    """Test saving MIDI data from base64"""
    from app.reference.mcp.midi_mate.main import save_midi_from_base64

    # Create a simple MIDI file in memory and encode it
    # This is a minimal MIDI header
    midi_bytes = b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60MTrk\x00\x00\x00\x04\x00\xff\x2f\x00"
    base64_data = base64.b64encode(midi_bytes).decode("utf-8")

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("app.reference.mcp.midi_mate.main.MidiFile") as mock_midi:
            mock_midi_instance = MagicMock()
            mock_midi_instance.tracks = [MagicMock()]
            mock_midi.return_value = mock_midi_instance

            result = save_midi_from_base64(base64_data, "test.mid", tmpdir)

            assert "MIDI saved" in result
            assert "test.mid" in result


def test_save_midi_from_base64_adds_extension():
    """Test that .mid extension is added if missing"""
    from app.reference.mcp.midi_mate.main import save_midi_from_base64

    midi_bytes = b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60MTrk\x00\x00\x00\x04\x00\xff\x2f\x00"
    base64_data = base64.b64encode(midi_bytes).decode("utf-8")

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("app.reference.mcp.midi_mate.main.MidiFile") as mock_midi:
            mock_midi_instance = MagicMock()
            mock_midi_instance.tracks = []
            mock_midi.return_value = mock_midi_instance

            result = save_midi_from_base64(base64_data, "test", tmpdir)

            assert ".mid" in result


def test_save_midi_from_base64_error_handling():
    """Test error handling for invalid base64 data"""
    from app.reference.mcp.midi_mate.main import save_midi_from_base64

    with tempfile.TemporaryDirectory() as tmpdir:
        result = save_midi_from_base64("invalid_base64!", "test.mid", tmpdir)

        assert "Error" in result


def test_save_midi_from_base64_creates_directory():
    """Test that output directory is created if it doesn't exist"""
    from app.reference.mcp.midi_mate.main import save_midi_from_base64

    midi_bytes = b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60MTrk\x00\x00\x00\x04\x00\xff\x2f\x00"
    base64_data = base64.b64encode(midi_bytes).decode("utf-8")

    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = os.path.join(tmpdir, "new_subdir")

        with patch("app.reference.mcp.midi_mate.main.MidiFile") as mock_midi:
            mock_midi_instance = MagicMock()
            mock_midi_instance.tracks = []
            mock_midi.return_value = mock_midi_instance

            result = save_midi_from_base64(base64_data, "test.mid", new_dir)

            assert os.path.exists(new_dir)
            assert "MIDI saved" in result
