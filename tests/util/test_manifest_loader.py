from unittest.mock import mock_open, patch, MagicMock

import pytest
import yaml

from app.util.manifest_loader import (
    load_manifest,
    sample_reference_artists,
    get_my_reference_proposals,
    get_sounds_like_by_color,
)


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


class TestGetMyReferenceProposals:
    @patch("os.walk")
    @patch("app.util.manifest_loader.load_manifest")
    @patch("os.getenv")
    def test_get_my_reference_proposals_no_matches(
        self, mock_getenv, mock_load_manifest, mock_walk
    ):
        """Test when no manifests match the color."""
        mock_getenv.return_value = "/test/path"
        mock_walk.return_value = [("/test/path", [], ["test.yml"])]

        # Create a mock manifest with different color
        mock_manifest = MagicMock()
        mock_manifest.rainbow_color.mnemonic_character_value = "R"
        mock_load_manifest.return_value = mock_manifest

        result = get_my_reference_proposals("B")

        assert len(result.iterations) == 0

    @patch("os.walk")
    @patch("app.util.manifest_loader.load_manifest")
    @patch("os.getenv")
    @patch("app.util.manifest_loader.SongProposalIteration")
    def test_get_my_reference_proposals_with_matches(
        self, mock_iteration_class, mock_getenv, mock_load_manifest, mock_walk
    ):
        """Test when manifests match the color."""
        mock_getenv.return_value = "/test/path"
        mock_walk.return_value = [("/test/path", [], ["test1.yml", "test2.yml"])]

        # Create mock manifests
        mock_rainbow_color = MagicMock()
        mock_rainbow_color.mnemonic_character_value = "B"

        mock_manifest = MagicMock()
        mock_manifest.rainbow_color = mock_rainbow_color
        mock_manifest.manifest_id = "test_v001"
        mock_manifest.bpm = 120
        mock_manifest.tempo = "Moderate"
        mock_manifest.key = "C major"
        mock_manifest.title = "Test Song"
        mock_manifest.mood = ["Happy"]
        mock_manifest.genres = ["Rock"]
        mock_manifest.concept = "Test concept for this musical piece"

        mock_load_manifest.return_value = mock_manifest

        # Mock the SongProposalIteration instance
        mock_iteration = MagicMock()
        mock_iteration.iteration_id = "test_v001"
        mock_iteration.bpm = 120
        mock_iteration_class.return_value = mock_iteration

        result = get_my_reference_proposals("B")

        assert len(result.iterations) == 2
        assert result.iterations[0].iteration_id == "test_v001"
        assert result.iterations[0].bpm == 120

    @patch("os.walk")
    @patch("app.util.manifest_loader.load_manifest")
    @patch("os.getenv")
    @patch("app.util.manifest_loader.SongProposalIteration")
    def test_get_my_reference_proposals_with_exception(
        self, mock_iteration_class, mock_getenv, mock_load_manifest, mock_walk
    ):
        """Test handling of exceptions during manifest loading."""
        mock_getenv.return_value = "/test/path"
        mock_walk.return_value = [("/test/path", [], ["bad.yml", "good.yml"])]

        # First call raises exception, second succeeds
        mock_rainbow_color = MagicMock()
        mock_rainbow_color.mnemonic_character_value = "B"

        mock_manifest = MagicMock()
        mock_manifest.rainbow_color = mock_rainbow_color
        mock_manifest.manifest_id = "test_v001"
        mock_manifest.bpm = 120
        mock_manifest.tempo = "Moderate"
        mock_manifest.key = "C major"
        mock_manifest.title = "Test Song"
        mock_manifest.mood = ["Happy"]
        mock_manifest.genres = ["Rock"]
        mock_manifest.concept = "Test concept for this musical piece"

        mock_load_manifest.side_effect = [Exception("Bad file"), mock_manifest]

        # Mock the SongProposalIteration instance
        mock_iteration = MagicMock()
        mock_iteration.iteration_id = "test_v001"
        mock_iteration.bpm = 120
        mock_iteration_class.return_value = mock_iteration

        result = get_my_reference_proposals("B")

        # Should skip the bad file and process the good one
        assert len(result.iterations) == 1

    @patch("os.walk")
    @patch("os.getenv")
    def test_get_my_reference_proposals_no_yml_files(self, mock_getenv, mock_walk):
        """Test when no YAML files are found."""
        mock_getenv.return_value = "/test/path"
        mock_walk.return_value = [("/test/path", [], [])]

        result = get_my_reference_proposals("B")

        assert len(result.iterations) == 0


class TestGetSoundsLikeByColor:
    @patch("os.walk")
    @patch("app.util.manifest_loader.load_manifest")
    @patch("os.getenv")
    def test_get_sounds_like_by_color_no_matches(
        self, mock_getenv, mock_load_manifest, mock_walk
    ):
        """Test when no manifests match the color."""
        mock_getenv.return_value = "/test/path"
        mock_walk.return_value = [("/test/path", [], ["test.yml"])]

        mock_manifest = MagicMock()
        mock_manifest.rainbow_color.mnemonic_character_value = "R"
        mock_load_manifest.return_value = mock_manifest

        result = get_sounds_like_by_color("B")

        assert result == []

    @patch("os.walk")
    @patch("app.util.manifest_loader.load_manifest")
    @patch("os.getenv")
    def test_get_sounds_like_by_color_with_matches(
        self, mock_getenv, mock_load_manifest, mock_walk
    ):
        """Test when manifests match and have sounds_like entries."""
        mock_getenv.return_value = "/test/path"
        mock_walk.return_value = [("/test/path", [], ["test.yml"])]

        # Create mock sounds_like entries
        sound1 = MagicMock()
        sound1.name = "Artist A"
        sound2 = MagicMock()
        sound2.name = "Artist B"

        mock_manifest = MagicMock()
        mock_manifest.rainbow_color.mnemonic_character_value = "B"
        mock_manifest.sounds_like = [sound1, sound2]
        mock_load_manifest.return_value = mock_manifest

        result = get_sounds_like_by_color("B")

        assert len(result) == 2
        assert "Artist A" in result
        assert "Artist B" in result

    @patch("os.walk")
    @patch("app.util.manifest_loader.load_manifest")
    @patch("os.getenv")
    def test_get_sounds_like_by_color_with_exception(
        self, mock_getenv, mock_load_manifest, mock_walk
    ):
        """Test handling of exceptions during manifest loading."""
        mock_getenv.return_value = "/test/path"
        mock_walk.return_value = [("/test/path", [], ["bad.yml", "good.yml"])]

        # First call raises exception, second succeeds
        sound1 = MagicMock()
        sound1.name = "Artist A"

        mock_manifest = MagicMock()
        mock_manifest.rainbow_color.mnemonic_character_value = "B"
        mock_manifest.sounds_like = [sound1]

        mock_load_manifest.side_effect = [Exception("Bad file"), mock_manifest]

        result = get_sounds_like_by_color("B")

        # Should skip the bad file and process the good one
        assert len(result) == 1
        assert result[0] == "Artist A"

    @patch("os.walk")
    @patch("app.util.manifest_loader.load_manifest")
    @patch("os.getenv")
    def test_get_sounds_like_by_color_multiple_manifests(
        self, mock_getenv, mock_load_manifest, mock_walk
    ):
        """Test aggregating sounds_like from multiple manifests."""
        mock_getenv.return_value = "/test/path"
        mock_walk.return_value = [("/test/path", [], ["test1.yml", "test2.yml"])]

        # Create two manifests with different sounds_like
        sound1 = MagicMock()
        sound1.name = "Artist A"
        sound2 = MagicMock()
        sound2.name = "Artist B"
        sound3 = MagicMock()
        sound3.name = "Artist C"

        mock_manifest1 = MagicMock()
        mock_manifest1.rainbow_color.mnemonic_character_value = "B"
        mock_manifest1.sounds_like = [sound1, sound2]

        mock_manifest2 = MagicMock()
        mock_manifest2.rainbow_color.mnemonic_character_value = "B"
        mock_manifest2.sounds_like = [sound3]

        mock_load_manifest.side_effect = [mock_manifest1, mock_manifest2]

        result = get_sounds_like_by_color("B")

        assert len(result) == 3
        assert "Artist A" in result
        assert "Artist B" in result
        assert "Artist C" in result
