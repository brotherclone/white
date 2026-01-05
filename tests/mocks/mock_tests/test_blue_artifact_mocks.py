import yaml
import os


def test_alternate_timeline_mocks():
    """Test that alternate timeline mock file loads"""
    mock_path = os.path.join(
        os.path.dirname(__file__), "..", "alternate_timeline_artifact_mock.yml"
    )
    with open(mock_path, "r") as f:
        data = yaml.safe_load(f)
    assert data is not None


def test_quantum_tape_label_mocks():
    """Test that quantum tape label mock file loads"""
    mock_path = os.path.join(
        os.path.dirname(__file__), "..", "quantum_tape_label_mock.yml"
    )
    with open(mock_path, "r") as f:
        data = yaml.safe_load(f)
    assert data is not None
