from app.structures.artifacts.circle_jerk_interview_artifact import (
    CircleJerkInterviewArtifact,
)

# ToDo: Add more tests


def test_init():
    """Test basic CircleJerkInterviewArtifact instantiation"""
    artifact = CircleJerkInterviewArtifact(thread_id="test")
    assert artifact.thread_id == "test"
