"""Tests for timestamp audio extraction pipeline."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from app.structures.manifests.manifest_song_structure import ManifestSongStructure
from app.structures.music.core.duration import Duration
from app.util.timestamp_audio_extractor import (
    adjust_segment_to_structure,
    duration_to_seconds,
    find_nearest_structure_boundary,
    split_long_segment,
)
from app.util.timestamp_pipeline import (
    PipelineConfig,
    SegmentMetadata,
    process_track_directory,
    process_multiple_tracks,
    process_staged_raw_material,
)


class TestDurationConversion:
    """Test duration conversion utilities."""

    def test_duration_to_seconds_from_object(self):
        """Test converting Duration object to seconds."""
        duration = Duration(minutes=1, seconds=30.5)
        assert duration_to_seconds(duration) == 90.5

    def test_duration_to_seconds_from_string(self):
        """Test converting string timestamp to seconds."""
        result = duration_to_seconds("[01:30.500]")
        assert result == 90.5

    def test_duration_zero(self):
        """Test zero duration."""
        duration = Duration(minutes=0, seconds=0.0)
        assert duration_to_seconds(duration) == 0.0


class TestStructureBoundaryDetection:
    """Test structure boundary detection and alignment."""

    @pytest.fixture
    def sample_structure(self):
        """Create sample structure sections."""
        return [
            ManifestSongStructure(
                section_name="Verse 1",
                start_time="[00:00.000]",
                end_time="[00:30.000]",
                description="First verse",
            ),
            ManifestSongStructure(
                section_name="Chorus 1",
                start_time="[00:30.000]",
                end_time="[01:00.000]",
                description="First chorus",
            ),
            ManifestSongStructure(
                section_name="Verse 2",
                start_time="[01:00.000]",
                end_time="[01:30.000]",
                description="Second verse",
            ),
        ]

    def test_find_boundary_within_threshold(self, sample_structure):
        """Test finding boundary within threshold distance."""
        # Timestamp at 29.5s, boundary at 30.0s (0.5s distance)
        result = find_nearest_structure_boundary(
            29.5, sample_structure, threshold_seconds=1.0
        )

        assert result is not None
        boundary_time, section_name = result
        assert boundary_time == 30.0
        assert section_name in ["Verse 1", "Chorus 1"]

    def test_find_boundary_outside_threshold(self, sample_structure):
        """Test that boundaries outside threshold are not found."""
        # Timestamp at 15.0s, nearest boundary is at 0.0s or 30.0s (15s distance)
        result = find_nearest_structure_boundary(
            15.0, sample_structure, threshold_seconds=2.0
        )

        assert result is None

    def test_adjust_segment_to_structure(self, sample_structure):
        """Test segment adjustment to structure boundaries."""
        # Segment from 28.0s to 32.0s, should snap to 30.0s boundaries
        adjusted_start, adjusted_end, metadata = adjust_segment_to_structure(
            28.0, 32.0, sample_structure, threshold_seconds=2.5
        )

        assert adjusted_start == 30.0
        assert adjusted_end == 30.0
        assert "adjustments" in metadata
        assert len(metadata["adjustments"]) == 2  # Both start and end adjusted

    def test_no_adjustment_when_far_from_boundaries(self, sample_structure):
        """Test that segments far from boundaries are not adjusted."""
        # Segment from 10.0s to 20.0s, far from any boundary
        adjusted_start, adjusted_end, metadata = adjust_segment_to_structure(
            10.0, 20.0, sample_structure, threshold_seconds=2.0
        )

        assert adjusted_start == 10.0
        assert adjusted_end == 20.0
        assert len(metadata["adjustments"]) == 0


class TestLongSegmentSplitting:
    """Test maximum segment length and splitting logic."""

    def test_segment_within_max_length(self):
        """Test that short segments are not split."""
        segments = split_long_segment(0.0, 25.0, max_length_seconds=30.0)

        assert len(segments) == 1
        assert segments[0] == (0.0, 25.0)

    def test_segment_split_with_overlap(self):
        """Test splitting long segment with overlap."""
        # 60 second segment, max 30s, 2s overlap
        segments = split_long_segment(
            0.0, 60.0, max_length_seconds=30.0, overlap_seconds=2.0
        )

        assert len(segments) == 3
        assert segments[0] == (0.0, 30.0)
        assert segments[1] == (28.0, 58.0)  # 30 - 2 = 28 start, 28 + 30 = 58 end
        assert segments[2][0] == 56.0  # 58 - 2 = 56
        assert segments[2][1] == 60.0  # Capped at end

    def test_segment_split_exact_multiple(self):
        """Test splitting when duration is exact multiple of max length."""
        segments = split_long_segment(
            0.0, 60.0, max_length_seconds=30.0, overlap_seconds=0.0
        )

        assert len(segments) == 2
        assert segments[0] == (0.0, 30.0)
        assert segments[1] == (30.0, 60.0)


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_lrc_file_exists(self):
        """Verify that sample LRC files exist for testing."""
        lrc_path = Path("staged_raw_material/01_01/01_01.lrc")
        if lrc_path.exists():
            assert lrc_path.exists()
        else:
            pytest.skip("Sample LRC files not available in test environment")

    def test_manifest_loading(self):
        """Test loading manifest from staged raw material."""
        manifest_path = Path("staged_raw_material/01_01/01_01.yml")
        if manifest_path.exists():
            from app.util.manifest_loader import load_manifest

            manifest = load_manifest(str(manifest_path))
            assert manifest.manifest_id == "01_01"
            assert len(manifest.structure) > 0
            assert manifest.lrc_file is not None
        else:
            pytest.skip("Sample manifest not available in test environment")


@pytest.mark.skipif(
    not os.path.exists("staged_raw_material/01_01"),
    reason="Staged raw material not available",
)
class TestRealDataProcessing:
    """Tests using real staged raw material (skipped if not available)."""

    def test_process_single_track(self, tmp_path):
        """Test processing a real track directory."""
        from app.util.timestamp_pipeline import PipelineConfig, process_track_directory

        track_dir = "staged_raw_material/01_01"
        output_dir = str(tmp_path / "output")

        config = PipelineConfig(
            max_segment_length_seconds=30.0,
            structure_alignment_threshold_seconds=2.0,
            extract_midi=False,  # Skip MIDI for faster test
            output_metadata=True,
        )

        result = process_track_directory(track_dir, output_dir, config)

        assert result["success"] is True
        assert result["segments_extracted"] > 0
        assert result["output_directory"] == str(Path(output_dir) / "01_01")

        # Check that metadata file was created
        metadata_file = Path(result["metadata_file"])
        assert metadata_file.exists()


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.max_segment_length_seconds == 30.0
        assert config.structure_alignment_threshold_seconds == 2.0
        assert config.overlap_seconds == 2.0
        assert config.extract_midi is True
        assert config.output_metadata is True
        assert config.sample_rate is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            max_segment_length_seconds=60.0,
            structure_alignment_threshold_seconds=5.0,
            overlap_seconds=1.0,
            extract_midi=False,
            output_metadata=False,
            sample_rate=44100,
        )

        assert config.max_segment_length_seconds == 60.0
        assert config.structure_alignment_threshold_seconds == 5.0
        assert config.overlap_seconds == 1.0
        assert config.extract_midi is False
        assert config.output_metadata is False
        assert config.sample_rate == 44100


class TestSegmentMetadata:
    """Test SegmentMetadata dataclass."""

    def test_segment_metadata_creation(self):
        """Test creating segment metadata."""
        metadata = SegmentMetadata(
            segment_id="test_001",
            track_id="track_01",
            audio_file="/path/to/audio.wav",
            midi_files=["/path/to/midi.mid"],
            start_seconds=0.0,
            end_seconds=30.0,
            duration_seconds=30.0,
            lyric_text="Test lyrics",
            segment_type="lyric",
            structure_adjustments=["adjusted start"],
            original_start=0.5,
            original_end=30.5,
            lrc_line_number=1,
            is_sub_segment=False,
            sub_segment_info=None,
        )

        assert metadata.segment_id == "test_001"
        assert metadata.track_id == "track_01"
        assert metadata.duration_seconds == 30.0
        assert metadata.is_sub_segment is False


class TestProcessTrackDirectory:
    """Test process_track_directory function."""

    def test_process_track_no_manifest(self, tmp_path):
        """Test processing track directory with no manifest file."""
        result = process_track_directory(str(tmp_path), str(tmp_path / "output"))

        assert result["success"] is False
        assert "No manifest file found" in result["error"]

    @patch("app.util.timestamp_pipeline.load_manifest")
    def test_process_track_manifest_load_error(self, mock_load_manifest, tmp_path):
        """Test handling manifest load errors."""
        # Create a dummy manifest file
        yml_file = tmp_path / "test.yml"
        yml_file.touch()

        mock_load_manifest.side_effect = Exception("Load error")

        result = process_track_directory(str(tmp_path), str(tmp_path / "output"))

        assert result["success"] is False
        assert "Failed to load manifest" in result["error"]

    @patch("app.util.timestamp_pipeline.load_manifest")
    def test_process_track_lrc_not_found(self, mock_load_manifest, tmp_path):
        """Test when LRC file is not found."""
        yml_file = tmp_path / "test.yml"
        yml_file.touch()

        mock_manifest = Mock()
        mock_manifest.lrc_file = "test.lrc"
        mock_manifest.main_audio_file = "test.wav"
        mock_load_manifest.return_value = mock_manifest

        result = process_track_directory(str(tmp_path), str(tmp_path / "output"))

        assert result["success"] is False
        assert "LRC file not found" in result["error"]

    @patch("app.util.timestamp_pipeline.load_manifest")
    def test_process_track_audio_not_found(self, mock_load_manifest, tmp_path):
        """Test when main audio file is not found."""
        yml_file = tmp_path / "test.yml"
        yml_file.touch()
        lrc_file = tmp_path / "test.lrc"
        lrc_file.touch()

        mock_manifest = Mock()
        mock_manifest.lrc_file = "test.lrc"
        mock_manifest.main_audio_file = "test.wav"
        mock_load_manifest.return_value = mock_manifest

        result = process_track_directory(str(tmp_path), str(tmp_path / "output"))

        assert result["success"] is False
        assert "Main audio file not found" in result["error"]

    @patch("app.util.timestamp_pipeline.load_manifest")
    @patch("app.util.timestamp_pipeline.create_segment_specs_from_lrc")
    def test_process_track_no_segments_created(
        self, mock_create_specs, mock_load_manifest, tmp_path
    ):
        """Test when no segments are created from LRC."""
        yml_file = tmp_path / "test.yml"
        yml_file.touch()
        lrc_file = tmp_path / "test.lrc"
        lrc_file.touch()
        audio_file = tmp_path / "test.wav"
        audio_file.touch()

        mock_manifest = Mock()
        mock_manifest.lrc_file = "test.lrc"
        mock_manifest.main_audio_file = "test.wav"
        mock_load_manifest.return_value = mock_manifest

        mock_create_specs.return_value = []

        result = process_track_directory(str(tmp_path), str(tmp_path / "output"))

        assert result["success"] is True
        assert result["segments_extracted"] == 0
        assert "warning" in result


class TestProcessMultipleTracks:
    """Test process_multiple_tracks function."""

    @patch("app.util.timestamp_pipeline.process_track_directory")
    def test_process_multiple_tracks_all_success(self, mock_process_track, tmp_path):
        """Test processing multiple tracks successfully."""
        mock_process_track.return_value = {
            "success": True,
            "segments_extracted": 10,
            "total_duration_seconds": 100.0,
        }

        track_dirs = [str(tmp_path / "track1"), str(tmp_path / "track2")]
        result = process_multiple_tracks(track_dirs, str(tmp_path / "output"))

        assert result["total_tracks_processed"] == 2
        assert result["successful_tracks"] == 2
        assert result["failed_tracks"] == 0
        assert result["total_segments_extracted"] == 20
        assert result["total_duration_seconds"] == 200.0

    @patch("app.util.timestamp_pipeline.process_track_directory")
    def test_process_multiple_tracks_some_failures(self, mock_process_track, tmp_path):
        """Test processing with some track failures."""
        mock_process_track.side_effect = [
            {
                "success": True,
                "segments_extracted": 10,
                "total_duration_seconds": 100.0,
            },
            {"success": False, "error": "Test error"},
        ]

        track_dirs = [str(tmp_path / "track1"), str(tmp_path / "track2")]
        result = process_multiple_tracks(track_dirs, str(tmp_path / "output"))

        assert result["total_tracks_processed"] == 2
        assert result["successful_tracks"] == 1
        assert result["failed_tracks"] == 1
        assert result["total_segments_extracted"] == 10
        assert len(result["failed_track_details"]) == 1

    def test_process_multiple_tracks_empty_list(self, tmp_path):
        """Test processing empty track list."""
        result = process_multiple_tracks([], str(tmp_path / "output"))

        assert result["total_tracks_processed"] == 0
        assert result["total_segments_extracted"] == 0

    def test_process_multiple_tracks_default_config(self, tmp_path):
        """Test that default config is created when None is passed."""
        with patch(
            "app.util.timestamp_pipeline.process_track_directory"
        ) as mock_process:
            mock_process.return_value = {"success": True, "segments_extracted": 0}

            result = process_multiple_tracks(
                [str(tmp_path / "track1")], str(tmp_path / "output"), config=None
            )

            # Should not raise error and should complete
            assert result["total_tracks_processed"] == 1


class TestProcessStagedRawMaterial:
    """Test process_staged_raw_material function."""

    def test_process_staged_nonexistent_directory(self, tmp_path):
        """Test with non-existent staged directory."""
        result = process_staged_raw_material(
            str(tmp_path / "nonexistent"), str(tmp_path / "output")
        )

        assert result["success"] is False
        assert "Staged directory not found" in result["error"]

    @patch("app.util.timestamp_pipeline.process_multiple_tracks")
    def test_process_staged_with_tracks(self, mock_process_multiple, tmp_path):
        """Test processing staged directory with tracks."""
        # Create mock track directories
        staged_dir = tmp_path / "staged"
        staged_dir.mkdir()
        (staged_dir / "track1").mkdir()
        (staged_dir / "track2").mkdir()

        mock_process_multiple.return_value = {
            "total_tracks_processed": 2,
            "successful_tracks": 2,
        }

        result = process_staged_raw_material(str(staged_dir), str(tmp_path / "output"))

        # Should call process_multiple_tracks
        mock_process_multiple.assert_called_once()
        assert result["total_tracks_processed"] == 2

    @patch("app.util.timestamp_pipeline.process_multiple_tracks")
    def test_process_staged_with_filter(self, mock_process_multiple, tmp_path):
        """Test processing with track filter."""
        staged_dir = tmp_path / "staged"
        staged_dir.mkdir()
        (staged_dir / "08_01").mkdir()
        (staged_dir / "08_02").mkdir()
        (staged_dir / "09_01").mkdir()

        mock_process_multiple.return_value = {"total_tracks_processed": 2}

        # Should only process tracks matching 08_*
        call_args = mock_process_multiple.call_args
        track_dirs = call_args[0][0]
        assert len(track_dirs) == 2
        assert all("08_" in str(d) for d in track_dirs)

    def test_process_staged_no_tracks(self, tmp_path):
        """Test processing directory with no track subdirectories."""
        staged_dir = tmp_path / "staged"
        staged_dir.mkdir()
        # Create a file instead of directory
        (staged_dir / "file.txt").touch()

        result = process_staged_raw_material(str(staged_dir), str(tmp_path / "output"))

        assert result["success"] is False
        assert "No track directories found" in result["error"]

    @patch("app.util.timestamp_pipeline.process_multiple_tracks")
    def test_process_staged_default_config(self, mock_process_multiple, tmp_path):
        """Test that default config is used when None is passed."""
        staged_dir = tmp_path / "staged"
        staged_dir.mkdir()
        (staged_dir / "track1").mkdir()

        mock_process_multiple.return_value = {"total_tracks_processed": 1}

        result = process_staged_raw_material(
            str(staged_dir), str(tmp_path / "output"), config=None
        )

        # Should not raise error
        assert "total_tracks_processed" in result
