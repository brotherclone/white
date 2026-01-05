import yaml
import os
import pytest


def test_circle_jerk_interview():
    """Test that circle jerk interview mock file loads - skip if doesn't exist"""
    mock_path = os.path.join(
        os.path.dirname(__file__), "..", "circle_jerk_interview_mock.yml"
    )
    if not os.path.exists(mock_path):
        pytest.skip(f"Mock file not found: {mock_path}")
    with open(mock_path, "r") as f:
        data = yaml.safe_load(f)
    assert data is not None
