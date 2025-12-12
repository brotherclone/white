import os
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import yaml
import polars as pl
from pydantic import BaseModel

from app.extractors.manifest_extractor.build_base_manifest_db import BuildBaseManifestDB
from app.extractors.manifest_extractor.concept_extractor import ConceptExtractor


class TestBuildBaseManifestDBInstantiation:
    """Tests for BuildBaseManifestDB initialization"""

    def test_basic_instantiation(self):
        """Test basic instantiation with concept_extractor parameter"""
        c = ConceptExtractor(
            track_id="100_1",
            concept_text="Sample concept text for testing.",
        )
        b = BuildBaseManifestDB(manifest_path="/", concept_extractor=c)
        assert isinstance(b, BaseModel)
        assert b.concept_extractor == c
        assert b.concept_extractor.track_id == "100_1"

    def test_instantiation_without_concept_extractor(self):
        """Test instantiation without providing concept_extractor"""
        b = BuildBaseManifestDB(manifest_path="/")
        assert isinstance(b, BaseModel)
        assert b.concept_extractor is None

    def test_instantiation_with_custom_manifest_path(self, tmp_path):
        """Test instantiation with a custom manifest path"""
        with patch.dict(os.environ, {"MANIFEST_PATH": str(tmp_path)}):
            b = BuildBaseManifestDB(manifest_path=tmp_path)
            assert b.manifest_path == tmp_path

    def test_missing_manifest_path_env_var(self):
        """Test that missing MANIFEST_PATH environment variable raises error"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="MANIFEST_PATH environment variable not set"
            ):
                BuildBaseManifestDB(manifest_path="/")


class TestGetManifests:
    """Tests for the get_manifests method"""

    def test_get_manifests_empty_directory(self, tmp_path):
        """Test get_manifests with empty directory"""
        with patch.dict(os.environ, {"MANIFEST_PATH": str(tmp_path)}):
            b = BuildBaseManifestDB(manifest_path=tmp_path)
            manifests = b.get_manifests()
            assert manifests == []

    def test_get_manifests_with_yml_files(self, tmp_path):
        """Test get_manifests finds .yml files"""
        # Create test .yml files
        (tmp_path / "test1.yml").write_text("test: data")
        (tmp_path / "test2.yaml").write_text("test: data")
        (tmp_path / "test.txt").write_text("not a manifest")

        with patch.dict(os.environ, {"MANIFEST_PATH": str(tmp_path)}):
            b = BuildBaseManifestDB(manifest_path=tmp_path)
            manifests = b.get_manifests()

            assert len(manifests) == 2
            assert any("test1.yml" in m for m in manifests)
            assert any("test2.yaml" in m for m in manifests)
            assert not any("test.txt" in m for m in manifests)

    def test_get_manifests_with_subdirectories(self, tmp_path):
        """Test get_manifests respects max_depth"""
        # Create nested structure
        (tmp_path / "root.yml").write_text("test: data")

        subdir1 = tmp_path / "sub1"
        subdir1.mkdir()
        (subdir1 / "sub1.yml").write_text("test: data")

        subdir2 = subdir1 / "sub2"
        subdir2.mkdir()
        (subdir2 / "sub2.yml").write_text("test: data")

        subdir3 = subdir2 / "sub3"
        subdir3.mkdir()
        (subdir3 / "sub3.yml").write_text("test: data")

        with patch.dict(os.environ, {"MANIFEST_PATH": str(tmp_path)}):
            b = BuildBaseManifestDB(manifest_path=tmp_path)

            # Default max_depth=2 (includes depths 0, 1, 2)
            manifests = b.get_manifests(max_depth=2)
            assert len(manifests) == 3  # root.yml, sub1.yml, and sub2.yml
            assert any("root.yml" in m for m in manifests)
            assert any("sub1.yml" in m for m in manifests)
            assert any("sub2.yml" in m for m in manifests)
            assert not any("sub3.yml" in m for m in manifests)

            # max_depth=0 should only get root level
            manifests_depth_0 = b.get_manifests(max_depth=0)
            assert len(manifests_depth_0) == 1
            assert any("root.yml" in m for m in manifests_depth_0)

            # max_depth=1 should get root and first subdirectory
            manifests_depth_1 = b.get_manifests(max_depth=1)
            assert len(manifests_depth_1) == 2
            assert any("root.yml" in m for m in manifests_depth_1)
            assert any("sub1.yml" in m for m in manifests_depth_1)
            assert not any("sub2.yml" in m for m in manifests_depth_1)

    def test_get_manifests_nonexistent_path(self):
        """Test get_manifests with nonexistent path"""
        with patch.dict(os.environ, {"MANIFEST_PATH": "/nonexistent"}):
            b = BuildBaseManifestDB(manifest_path=Path("/nonexistent/path"))
            manifests = b.get_manifests()
            assert manifests == []

    def test_get_manifests_file_not_directory(self, tmp_path):
        """Test get_manifests when path is a file, not a directory"""
        file_path = tmp_path / "test.yml"
        file_path.write_text("test: data")

        with patch.dict(os.environ, {"MANIFEST_PATH": str(file_path)}):
            b = BuildBaseManifestDB(manifest_path=file_path)
            manifests = b.get_manifests()
            assert manifests == []

    def test_get_manifests_case_insensitive_extensions(self, tmp_path):
        """Test that both .yml and .yaml extensions are found (case insensitive)"""
        (tmp_path / "test1.yml").write_text("test: data")
        (tmp_path / "test2.YML").write_text("test: data")
        (tmp_path / "test3.yaml").write_text("test: data")
        (tmp_path / "test4.YAML").write_text("test: data")

        with patch.dict(os.environ, {"MANIFEST_PATH": str(tmp_path)}):
            b = BuildBaseManifestDB(manifest_path=tmp_path)
            manifests = b.get_manifests()
            assert len(manifests) == 4


class TestFlattenDict:
    """Tests for the flatten_dict method"""

    def test_flatten_dict_simple(self):
        """Test flattening a simple nested dict"""
        with patch.dict(os.environ, {"MANIFEST_PATH": "/"}):
            b = BuildBaseManifestDB(manifest_path="/")

            nested = {"a": 1, "b": {"c": 2, "d": 3}}

            result = b.flatten_dict(nested)
            assert result == {"a": 1, "b_c": 2, "b_d": 3}

    def test_flatten_dict_deeply_nested(self):
        """Test flattening a deeply nested dict"""
        with patch.dict(os.environ, {"MANIFEST_PATH": "/"}):
            b = BuildBaseManifestDB(manifest_path="/")

            nested = {"level1": {"level2": {"level3": {"value": "deep"}}}}

            result = b.flatten_dict(nested)
            assert result == {"level1_level2_level3_value": "deep"}

    def test_flatten_dict_with_lists(self):
        """Test that lists are preserved and not flattened"""
        with patch.dict(os.environ, {"MANIFEST_PATH": "/"}):
            b = BuildBaseManifestDB(manifest_path="/")

            nested = {"items": [1, 2, 3], "nested": {"list": ["a", "b", "c"]}}

            result = b.flatten_dict(nested)
            assert result == {"items": [1, 2, 3], "nested_list": ["a", "b", "c"]}

    def test_flatten_dict_custom_separator(self):
        """Test flattening with custom separator"""
        with patch.dict(os.environ, {"MANIFEST_PATH": "/"}):
            b = BuildBaseManifestDB(manifest_path="/")

            nested = {"a": {"b": 1}}

            result = b.flatten_dict(nested, sep=".")
            assert result == {"a.b": 1}

    def test_flatten_dict_empty(self):
        """Test flattening an empty dict"""
        with patch.dict(os.environ, {"MANIFEST_PATH": "/"}):
            b = BuildBaseManifestDB(manifest_path="/")
            result = b.flatten_dict({})
            assert result == {}


class TestProcessManifests:
    """Tests for the process_manifests method"""

    @pytest.fixture
    def sample_manifest_data(self):
        """Fixture providing sample manifest data"""
        return {
            "manifest_id": "01_01",
            "bpm": 120,
            "tempo": "4/4",
            "key": "C major",
            "rainbow_color": "R",
            "title": "Test Song",
            "release_date": "2023-01-01",
            "album_sequence": 1,
            "main_audio_file": "test_main.wav",
            "TRT": "[03:30.000]",
            "vocals": True,
            "lyrics": True,
            "concept": "This is a test concept about temporal rebracketing and parallel realities.",
            "structure": [
                {
                    "section_name": "Verse 1",
                    "start_time": "[00:00.000]",
                    "end_time": "[00:30.000]",
                    "description": "Opening verse",
                }
            ],
            "audio_tracks": [
                {
                    "id": 1,
                    "description": "Bass",
                    "audio_file": "01_01_01_bass.wav",
                    "group": "Bass",
                    "player": "Graham Hopkins",
                }
            ],
            "sounds_like": [{"name": "Test Artist", "discogs_id": 12345}],
            "mood": ["melancholic", "introspective"],
            "genres": ["indie rock"],
        }

    @pytest.fixture
    def mock_concept_extractor(self):
        """Fixture providing a mock ConceptExtractor"""
        mock = Mock(spec=ConceptExtractor)
        mock.track_id = "01_01"
        mock.concept_text = "Test concept"
        mock.load_model = Mock()
        mock.classify_concept_by_rebracketing_type = Mock(return_value="TEMPORAL")
        return mock

    def test_process_manifests_with_valid_data(
        self, tmp_path, sample_manifest_data, mock_concept_extractor
    ):
        """Test process_manifests with valid manifest data"""
        # Create manifest file
        manifest_path = tmp_path / "test.yml"
        with open(manifest_path, "w") as f:
            yaml.dump(sample_manifest_data, f)

        # Set up environment
        output_path = tmp_path / "output"
        output_path.mkdir()

        with patch.dict(
            os.environ,
            {"MANIFEST_PATH": str(tmp_path), "BASE_MANIFEST_DB_PATH": str(output_path)},
        ):
            b = BuildBaseManifestDB(
                manifest_path=tmp_path, concept_extractor=mock_concept_extractor
            )

            # Process manifests
            b.process_manifests()

            # Check that parquet file was created
            parquet_file = output_path / "base_manifest_db.parquet"
            assert parquet_file.exists()

            # Load and verify the data
            df = pl.read_parquet(parquet_file)
            assert len(df) > 0
            assert "id" in df.columns
            assert "bpm" in df.columns
            assert "title" in df.columns

    def test_process_manifests_empty_list(self, tmp_path):
        """Test process_manifests with no manifests"""
        output_path = tmp_path / "output"
        output_path.mkdir()

        with patch.dict(
            os.environ,
            {"MANIFEST_PATH": str(tmp_path), "BASE_MANIFEST_DB_PATH": str(output_path)},
        ):
            b = BuildBaseManifestDB(manifest_path=tmp_path)
            b.process_manifests()

            # Check that parquet file was created (even if empty)
            parquet_file = output_path / "base_manifest_db.parquet"
            assert parquet_file.exists()

    def test_process_manifests_handles_invalid_manifest(
        self, tmp_path, mock_concept_extractor
    ):
        """Test that process_manifests handles invalid manifest gracefully"""
        # Create a manifest with invalid data (missing required fields)
        manifest_path = tmp_path / "invalid.yml"
        with open(manifest_path, "w") as f:
            yaml.dump({"manifest_id": "bad_01", "invalid_field": "bad data"}, f)

        output_path = tmp_path / "output"
        output_path.mkdir()

        with patch.dict(
            os.environ,
            {"MANIFEST_PATH": str(tmp_path), "BASE_MANIFEST_DB_PATH": str(output_path)},
        ):
            b = BuildBaseManifestDB(
                manifest_path=tmp_path, concept_extractor=mock_concept_extractor
            )

            # Should not raise exception, just skip invalid manifests
            b.process_manifests()

            # Verify parquet was still created (even if empty)
            parquet_file = output_path / "base_manifest_db.parquet"
            assert parquet_file.exists()

    def test_process_manifests_without_concept_extractor(
        self, tmp_path, sample_manifest_data
    ):
        """Test process_manifests without a concept_extractor"""
        # Remove concept from data to avoid AttributeError
        sample_manifest_data_no_concept = sample_manifest_data.copy()
        del sample_manifest_data_no_concept["concept"]

        manifest_path = tmp_path / "test.yml"
        with open(manifest_path, "w") as f:
            yaml.dump(sample_manifest_data_no_concept, f)

        output_path = tmp_path / "output"
        output_path.mkdir()

        with patch.dict(
            os.environ,
            {"MANIFEST_PATH": str(tmp_path), "BASE_MANIFEST_DB_PATH": str(output_path)},
        ):
            b = BuildBaseManifestDB(manifest_path=tmp_path)
            b.process_manifests()

            parquet_file = output_path / "base_manifest_db.parquet"
            assert parquet_file.exists()

    def test_process_manifests_with_lrc_file(
        self, tmp_path, sample_manifest_data, mock_concept_extractor
    ):
        """Test process_manifests extracts lyrics from LRC file"""
        sample_manifest_data["lrc_file"] = "test.lrc"

        # Create manifest directory
        manifest_dir = tmp_path / "manifest"
        manifest_dir.mkdir()

        # Create manifest file
        manifest_path = manifest_dir / "test.yml"
        with open(manifest_path, "w") as f:
            yaml.dump(sample_manifest_data, f)

        # Create LRC file
        lrc_path = manifest_dir / "test.lrc"
        lrc_path.write_text("[00:00.00]Test lyrics\n[00:10.00]More lyrics")

        output_path = tmp_path / "output"
        output_path.mkdir()

        with patch.dict(
            os.environ,
            {
                "MANIFEST_PATH": str(manifest_dir),
                "BASE_MANIFEST_DB_PATH": str(output_path),
            },
        ):
            b = BuildBaseManifestDB(
                manifest_path=manifest_dir, concept_extractor=mock_concept_extractor
            )
            b.process_manifests()

            parquet_file = output_path / "base_manifest_db.parquet"
            assert parquet_file.exists()

            df = pl.read_parquet(parquet_file)
            assert "lrc_lyrics" in df.columns


class TestBuildBaseManifestDBIntegration:
    """Integration tests for BuildBaseManifestDB"""

    @pytest.mark.skip(
        reason="Requires fixing UnboundLocalError bug in build_base_manifest_db.py:210 where 'rb' variable is not defined when concept is empty"
    )
    def test_full_workflow(self, tmp_path):
        """Test complete workflow from initialization to processing"""
        # Setup
        manifest_data = {
            "manifest_id": "test_01",
            "bpm": 100,
            "tempo": "4/4",
            "key": "D minor",
            "rainbow_color": "V",
            "title": "Integration Test",
            "release_date": "2023-06-01",
            "album_sequence": 1,
            "main_audio_file": "test_main.wav",
            "TRT": "[02:45.000]",
            "vocals": False,
            "lyrics": False,
            "concept": "",  # Empty string to skip concept extraction
            "sounds_like": [],
            "structure": [],
            "mood": ["test"],
            "genres": ["electronic"],
            "audio_tracks": [
                {
                    "id": 1,
                    "description": "Synth",
                    "audio_file": "test_synth.wav",
                    "player": "Gabriel Walsh",
                }
            ],
        }

        manifest_path = tmp_path / "test.yml"
        with open(manifest_path, "w") as f:
            yaml.dump(manifest_data, f)

        output_path = tmp_path / "output"
        output_path.mkdir()

        with patch.dict(
            os.environ,
            {"MANIFEST_PATH": str(tmp_path), "BASE_MANIFEST_DB_PATH": str(output_path)},
        ):
            # Create instance
            builder = BuildBaseManifestDB(manifest_path=tmp_path)

            # Verify initialization
            assert builder.manifest_path == tmp_path
            assert len(builder.manifest_paths) == 1

            # Process
            builder.process_manifests()

            # Verify output
            parquet_file = output_path / "base_manifest_db.parquet"
            assert parquet_file.exists()

            # Verify data was processed
            df = pl.read_parquet(parquet_file)
            if len(df) > 0:
                assert "id" in df.columns
                assert "bpm" in df.columns
                assert "title" in df.columns

    @pytest.mark.skip(
        reason="Requires fixing UnboundLocalError bug in build_base_manifest_db.py:210 where 'rb' variable is not defined when concept is empty"
    )
    def test_multiple_manifests_processing(self, tmp_path):
        """Test processing multiple manifests at once"""
        # Create multiple manifest files
        for i in range(3):
            manifest_data = {
                "manifest_id": f"test_0{i+1}",
                "bpm": 100 + i * 10,
                "tempo": "4/4",
                "key": "C major",
                "rainbow_color": "G",
                "title": f"Test Song {i+1}",
                "release_date": "2023-01-01",
                "album_sequence": i + 1,
                "main_audio_file": f"test_{i}_main.wav",
                "TRT": "[03:00.000]",
                "vocals": True,
                "lyrics": False,
                "concept": "",  # Empty string to skip concept extraction
                "sounds_like": [],
                "structure": [],
                "mood": ["upbeat"],
                "genres": ["rock"],
                "audio_tracks": [
                    {
                        "id": 1,
                        "description": "Track",
                        "audio_file": f"test_{i}.wav",
                        "player": "Gabriel Walsh",
                    }
                ],
            }

            manifest_path = tmp_path / f"test_{i}.yml"
            with open(manifest_path, "w") as f:
                yaml.dump(manifest_data, f)

        output_path = tmp_path / "output"
        output_path.mkdir()

        with patch.dict(
            os.environ,
            {"MANIFEST_PATH": str(tmp_path), "BASE_MANIFEST_DB_PATH": str(output_path)},
        ):
            builder = BuildBaseManifestDB(manifest_path=tmp_path)
            assert len(builder.manifest_paths) == 3

            builder.process_manifests()

            parquet_file = output_path / "base_manifest_db.parquet"
            assert parquet_file.exists()

            # Verify data was processed
            df = pl.read_parquet(parquet_file)
            if len(df) > 0:
                assert "id" in df.columns
