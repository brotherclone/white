"""Tests for MIDI segmentation utilities."""

from unittest.mock import Mock, patch, MagicMock

import pytest

from app.util import midi_segment_utils


# Test MIDO_AVAILABLE flag
def test_mido_available_flag():
    """Test that MIDO_AVAILABLE flag is a boolean."""
    assert isinstance(midi_segment_utils.MIDO_AVAILABLE, bool)


# Test when mido is not available
@pytest.mark.skipif(midi_segment_utils.MIDO_AVAILABLE, reason="mido is available")
def test_segment_midi_file_without_mido():
    """Test segment_midi_file fails gracefully when mido is not available."""
    result = midi_segment_utils.segment_midi_file("dummy.mid", 0.0, 10.0, "output.mid")
    assert result is False


@pytest.mark.skipif(midi_segment_utils.MIDO_AVAILABLE, reason="mido is available")
def test_segment_midi_with_audio_without_mido():
    """Test segment_midi_with_audio returns empty list when mido is not available."""
    result = midi_segment_utils.segment_midi_with_audio(
        "dummy.wav", 0.0, 10.0, "/tmp", []
    )
    assert result == []


@pytest.mark.skipif(midi_segment_utils.MIDO_AVAILABLE, reason="mido is available")
def test_extract_midi_note_density_without_mido():
    """Test extract_midi_note_density returns None when mido is not available."""
    result = midi_segment_utils.extract_midi_note_density("dummy.mid")
    assert result is None


# Tests when mido IS available
@pytest.mark.skipif(not midi_segment_utils.MIDO_AVAILABLE, reason="mido not available")
class TestWithMido:
    """Tests that require mido to be available."""

    def test_segment_midi_file_with_nonexistent_file(self, tmp_path):
        """Test segment_midi_file with a nonexistent file."""
        output_path = str(tmp_path / "output.mid")
        result = midi_segment_utils.segment_midi_file(
            "nonexistent.mid", 0.0, 10.0, output_path
        )
        assert result is False

    @patch("app.util.midi_segment_utils.MidiFile")
    def test_segment_midi_file_with_exception(self, mock_midifile, tmp_path):
        """Test segment_midi_file handles exceptions."""
        mock_midifile.side_effect = Exception("Test error")
        output_path = str(tmp_path / "output.mid")

        result = midi_segment_utils.segment_midi_file(
            "test.mid", 0.0, 10.0, output_path
        )

        assert result is False

    @patch("app.util.midi_segment_utils.MidiFile")
    @patch("os.makedirs")
    def test_segment_midi_file_creates_output_directory(
        self, mock_makedirs, mock_midifile, tmp_path
    ):
        """Test segment_midi_file creates output directory."""
        # Setup mock MIDI file
        mock_midi = MagicMock()
        mock_midi.type = 1
        mock_midi.ticks_per_beat = 480
        mock_midi.tracks = []
        mock_midifile.return_value = mock_midi

        output_path = str(tmp_path / "subdir" / "output.mid")

        midi_segment_utils.segment_midi_file("test.mid", 0.0, 10.0, output_path)

        # Verify makedirs was called
        mock_makedirs.assert_called()

    def test_find_matching_midi_files_with_audio_file(self, tmp_path):
        """Test find_matching_midi_files with matching audio files."""
        # Create test directory structure
        audio_file = tmp_path / "test_audio.wav"
        midi_file = tmp_path / "test_midi.mid"

        audio_file.touch()
        midi_file.touch()

        # Mock manifest track
        mock_track = Mock()
        mock_track.audio_file = "test_audio.wav"
        mock_track.midi_file = "test_midi.mid"
        mock_track.midi_group_file = None

        result = midi_segment_utils.find_matching_midi_files(
            str(audio_file), [mock_track]
        )

        assert len(result) == 1
        assert result[0] == str(midi_file)

    def test_find_matching_midi_files_with_midi_group_file(self, tmp_path):
        """Test find_matching_midi_files with midi_group_file."""
        # Create test directory structure
        audio_file = tmp_path / "test_audio.wav"
        midi_file = tmp_path / "test_group.mid"

        audio_file.touch()
        midi_file.touch()

        # Mock manifest track
        mock_track = Mock()
        mock_track.audio_file = "test_audio.wav"
        mock_track.midi_file = None
        mock_track.midi_group_file = "test_group.mid"

        result = midi_segment_utils.find_matching_midi_files(
            str(audio_file), [mock_track]
        )

        assert len(result) == 1
        assert result[0] == str(midi_file)

    def test_find_matching_midi_files_with_both_types(self, tmp_path):
        """Test find_matching_midi_files with both midi_file and midi_group_file."""
        # Create test directory structure
        audio_file = tmp_path / "test_audio.wav"
        midi_file1 = tmp_path / "test_midi.mid"
        midi_file2 = tmp_path / "test_group.mid"

        audio_file.touch()
        midi_file1.touch()
        midi_file2.touch()

        # Mock manifest track
        mock_track = Mock()
        mock_track.audio_file = "test_audio.wav"
        mock_track.midi_file = "test_midi.mid"
        mock_track.midi_group_file = "test_group.mid"

        result = midi_segment_utils.find_matching_midi_files(
            str(audio_file), [mock_track]
        )

        assert len(result) == 2

    def test_find_matching_midi_files_no_match(self, tmp_path):
        """Test find_matching_midi_files with no matching files."""
        audio_file = tmp_path / "test_audio.wav"
        audio_file.touch()

        # Mock manifest track with different audio file
        mock_track = Mock()
        mock_track.audio_file = "other_audio.wav"
        mock_track.midi_file = "test_midi.mid"

        result = midi_segment_utils.find_matching_midi_files(
            str(audio_file), [mock_track]
        )

        assert len(result) == 0

    def test_find_matching_midi_files_nonexistent_midi(self, tmp_path):
        """Test find_matching_midi_files when MIDI file doesn't exist."""
        audio_file = tmp_path / "test_audio.wav"
        audio_file.touch()

        # Mock manifest track pointing to nonexistent MIDI
        mock_track = Mock()
        mock_track.audio_file = "test_audio.wav"
        mock_track.midi_file = "nonexistent.mid"
        mock_track.midi_group_file = None

        result = midi_segment_utils.find_matching_midi_files(
            str(audio_file), [mock_track]
        )

        assert len(result) == 0

    def test_segment_midi_with_audio_no_manifest(self, tmp_path):
        """Test segment_midi_with_audio without manifest (fallback mode)."""
        # Create test files
        audio_file = tmp_path / "test_audio.wav"
        midi_file = tmp_path / "test_audio_midi.mid"
        output_dir = tmp_path / "output"

        audio_file.touch()
        midi_file.touch()
        output_dir.mkdir()

        with patch("app.util.midi_segment_utils.segment_midi_file") as mock_segment:
            mock_segment.return_value = True
            mock_segment.assert_called()

    def test_segment_midi_with_audio_with_manifest(self, tmp_path):
        """Test segment_midi_with_audio with manifest tracks."""
        # Create test files
        audio_file = tmp_path / "test_audio.wav"
        midi_file = tmp_path / "test_midi.mid"
        output_dir = tmp_path / "output"

        audio_file.touch()
        midi_file.touch()
        output_dir.mkdir()

        # Mock manifest track
        mock_track = Mock()
        mock_track.audio_file = "test_audio.wav"
        mock_track.midi_file = "test_midi.mid"
        mock_track.midi_group_file = None

        with patch("app.util.midi_segment_utils.segment_midi_file") as mock_segment:
            mock_segment.return_value = True

            result = midi_segment_utils.segment_midi_with_audio(
                str(audio_file), 0.0, 10.0, str(output_dir), [mock_track]
            )

            assert len(result) > 0
            mock_segment.assert_called()

    def test_segment_midi_with_audio_no_midi_files(self, tmp_path):
        """Test segment_midi_with_audio when no MIDI files found."""
        audio_file = tmp_path / "test_audio.wav"
        output_dir = tmp_path / "output"

        audio_file.touch()
        output_dir.mkdir()

        result = midi_segment_utils.segment_midi_with_audio(
            str(audio_file), 0.0, 10.0, str(output_dir), []
        )

        assert result == []

    def test_segment_midi_with_audio_segment_fails(self, tmp_path):
        """Test segment_midi_with_audio when segmentation fails."""
        audio_file = tmp_path / "test_audio.wav"
        midi_file = tmp_path / "test_midi.mid"
        output_dir = tmp_path / "output"

        audio_file.touch()
        midi_file.touch()
        output_dir.mkdir()

        mock_track = Mock()
        mock_track.audio_file = "test_audio.wav"
        mock_track.midi_file = "test_midi.mid"
        mock_track.midi_group_file = None

        with patch("app.util.midi_segment_utils.segment_midi_file") as mock_segment:
            mock_segment.return_value = False

            result = midi_segment_utils.segment_midi_with_audio(
                str(audio_file), 0.0, 10.0, str(output_dir), [mock_track]
            )

            # Should return empty list when segmentation fails
            assert result == []

    @patch("app.util.midi_segment_utils.MidiFile")
    def test_extract_midi_note_density_success(self, mock_midifile):
        """Test extract_midi_note_density with valid MIDI file."""
        # Mock MIDI file with note events
        mock_track = MagicMock()
        mock_msg1 = Mock(type="note_on", velocity=64, time=100)
        mock_msg2 = Mock(type="note_on", velocity=80, time=100)
        mock_msg3 = Mock(type="note_off", velocity=0, time=100)
        mock_track.__iter__ = Mock(return_value=iter([mock_msg1, mock_msg2, mock_msg3]))

        mock_midi = MagicMock()
        mock_midi.ticks_per_beat = 480
        mock_midi.tracks = [mock_track]
        mock_midifile.return_value = mock_midi

        with patch("mido.tick2second", return_value=0.1):
            result = midi_segment_utils.extract_midi_note_density("test.mid")

            assert result is not None
            assert result >= 0

    @patch("app.util.midi_segment_utils.MidiFile")
    def test_extract_midi_note_density_empty_file(self, mock_midifile):
        """Test extract_midi_note_density with empty MIDI file."""
        mock_track = MagicMock()
        mock_track.__iter__ = Mock(return_value=iter([]))

        mock_midi = MagicMock()
        mock_midi.ticks_per_beat = 480
        mock_midi.tracks = [mock_track]
        mock_midifile.return_value = mock_midi

        result = midi_segment_utils.extract_midi_note_density("test.mid")

        assert result == 0.0

    @patch("app.util.midi_segment_utils.MidiFile")
    def test_extract_midi_note_density_exception(self, mock_midifile):
        """Test extract_midi_note_density handles exceptions."""
        mock_midifile.side_effect = Exception("Test error")

        result = midi_segment_utils.extract_midi_note_density("test.mid")

        assert result is None
