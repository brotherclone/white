import yaml
import os
import pytest


def test_arbitrarys_survey_mocks():
    """Test that arbitrarys survey mock file loads"""
    mock_path = os.path.join(
        os.path.dirname(__file__), "..", "arbitrarys_survey_mock.yml"
    )
    with open(mock_path, "r") as f:
        data = yaml.safe_load(f)
    assert data is not None


def test_last_human_mocks():
    """Test that last human mock file loads - skip if doesn't exist"""
    mock_path = os.path.join(os.path.dirname(__file__), "..", "last_human_mock.yml")
    if not os.path.exists(mock_path):
        pytest.skip(f"Mock file not found: {mock_path}")
    with open(mock_path, "r") as f:
        data = yaml.safe_load(f)
    assert data is not None


def test_species_extinction_mocks():
    """Test that species extinction mock file loads - skip if doesn't exist"""
    mock_path = os.path.join(
        os.path.dirname(__file__), "..", "species_extinction_mock.yml"
    )
    if not os.path.exists(mock_path):
        pytest.skip(f"Mock file not found: {mock_path}")
    with open(mock_path, "r") as f:
        data = yaml.safe_load(f)
    assert data is not None


def test_rescue_decision_mocks():
    """Test that rescue decision mock file loads"""
    mock_path = os.path.join(
        os.path.dirname(__file__), "..", "rescue_decision_mock.yml"
    )
    with open(mock_path, "r") as f:
        data = yaml.safe_load(f)
    assert data is not None


def test_species_extinction_narrative_mocks():
    """Test that species extinction narrative mock file loads - skip if doesn't exist"""
    mock_path = os.path.join(
        os.path.dirname(__file__), "..", "species_extinction_narrative_mock.yml"
    )
    if not os.path.exists(mock_path):
        pytest.skip(f"Mock file not found: {mock_path}")
    with open(mock_path, "r") as f:
        data = yaml.safe_load(f)
    assert data is not None
