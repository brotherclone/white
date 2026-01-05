from unittest.mock import mock_open, patch

import pytest
import yaml

from app.util.manifest_loader import load_manifest, sample_reference_artists


def test_load_manifest_success():
    yaml_content = "title: Test Song\nartist: Test Artist"
    with (
        patch("builtins.open", mock_open(read_data=yaml_content)) as mock_file,
        patch("app.util.manifest_loader.Manifest") as mock_manifest,
    ):
        mock_manifest.return_value = "manifest_obj"
        result = load_manifest("dummy_path.yml")
        mock_file.assert_called_once_with("dummy_path.yml", "r")
        mock_manifest.assert_called_once_with(title="Test Song", artist="Test Artist")
        assert result == "manifest_obj"


def test_load_manifest_file_not_found():
    with patch("builtins.open", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            load_manifest("missing.yml")


def test_load_manifest_invalid_yaml():
    with (
        patch("builtins.open", mock_open(read_data=": invalid yaml")),
        patch("app.util.manifest_loader.Manifest"),
    ):
        with pytest.raises(yaml.YAMLError):
            load_manifest("bad.yml")


def test_load_manifest_invalid_data():
    yaml_content = "not_a_field: value"
    with (
        patch("builtins.open", mock_open(read_data=yaml_content)),
        patch(
            "app.util.manifest_loader.Manifest",
            side_effect=Exception("validation error"),
        ),
    ):
        with pytest.raises(Exception) as excinfo:
            load_manifest("invalid_data.yml")
        assert "validation error" in str(excinfo.value)


class TestSampleReferenceArtists:
    def test_empty_list_returns_empty(self):
        result = sample_reference_artists([])
        assert result == []

    def test_none_input_returns_empty(self):
        result = sample_reference_artists(None)
        assert result == []

    def test_single_artist_returns_one_artist(self):
        artists = ["Artist A"]
        result = sample_reference_artists(artists)
        assert len(result) == 1
        assert result[0] == "Artist A"

    def test_two_artists_returns_two_artists(self):
        artists = ["Artist A", "Artist B"]
        result = sample_reference_artists(artists, min_k=2, max_k=2)
        assert len(result) == 2
        assert set(result) == {"Artist A", "Artist B"}

    def test_returns_between_min_and_max(self):
        artists = ["A", "B", "C", "D", "E", "F"]
        result = sample_reference_artists(artists, min_k=2, max_k=4)
        assert 2 <= len(result) <= 4
        assert all(artist in artists for artist in result)

    def test_no_duplicates_by_default(self):
        artists = ["A", "B", "C", "D", "E"]
        result = sample_reference_artists(artists, min_k=3, max_k=4)
        # Check that all items in result are unique
        assert len(result) == len(set(result))

    def test_respects_max_artists_when_fewer_available(self):
        artists = ["A", "B"]
        result = sample_reference_artists(artists, min_k=5, max_k=10)
        # Should return at most 2 artists since that's all we have
        assert len(result) == 2
        assert set(result) == {"A", "B"}

    def test_allow_duplicates_true(self):
        artists = ["A", "B"]
        # With allow_duplicates=True and asking for more than available
        with patch("random.randint", return_value=5):
            result = sample_reference_artists(
                artists, min_k=5, max_k=5, allow_duplicates=True
            )
            assert len(result) == 5
            # May contain duplicates since we only have 2 artists but asked for 5

    def test_allow_duplicates_false_limits_to_list_size(self):
        artists = ["A", "B", "C"]
        with patch("random.randint", return_value=10):
            result = sample_reference_artists(
                artists, min_k=10, max_k=10, allow_duplicates=False
            )
            # Should be limited to 3 (the list size) since duplicates not allowed
            assert len(result) == 3
            assert len(set(result)) == 3  # All unique
