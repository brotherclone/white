import yaml
import os
import pytest


def test_audio_infranym_mocks():
    """Test that audio infranym mock file loads - skip if doesn't exist"""
    mock_path = os.path.join(os.path.dirname(__file__), "..", "infranym_audio_mock.yml")
    if not os.path.exists(mock_path):
        pytest.skip(f"Mock file not found: {mock_path}")
    with open(mock_path, "r") as f:
        data = yaml.safe_load(f)
    assert data is not None


def test_image_infranym_mocks():
    """Test that image infranym mock file loads - skip if doesn't exist"""
    mock_path = os.path.join(os.path.dirname(__file__), "..", "infranym_image_mock.yml")
    if not os.path.exists(mock_path):
        pytest.skip(f"Mock file not found: {mock_path}")
    with open(mock_path, "r") as f:
        data = yaml.safe_load(f)
    assert data is not None


def test_midi_infranym_mocks():
    """Test that MIDI infranym mock file loads - skip if doesn't exist"""
    mock_path = os.path.join(os.path.dirname(__file__), "..", "infranym_midi_mock.yml")
    if not os.path.exists(mock_path):
        pytest.skip(f"Mock file not found: {mock_path}")
    with open(mock_path, "r") as f:
        data = yaml.safe_load(f)
    assert data is not None


def test_text_infranym_mocks():
    """Test that text infranym mock file loads - skip if doesn't exist"""
    mock_path = os.path.join(os.path.dirname(__file__), "..", "infranym_text_mock.yml")
    if not os.path.exists(mock_path):
        pytest.skip(f"Mock file not found: {mock_path}")
    with open(mock_path, "r") as f:
        data = yaml.safe_load(f)
    assert data is not None
