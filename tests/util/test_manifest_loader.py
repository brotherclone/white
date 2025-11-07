from unittest.mock import mock_open, patch

import pytest
import yaml

from app.util.manifest_loader import load_manifest


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
