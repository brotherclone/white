from app.structures.artifacts.newspaper_artifact import NewspaperArtifact
from app.structures.enums.chain_artifact_type import ChainArtifactType


def test_newspaper_artifact_defaults():
    artifact = NewspaperArtifact(thread_id="thread-1")
    assert artifact.thread_id == "thread-1"
    assert artifact.chain_artifact_type == ChainArtifactType.NEWSPAPER_ARTICLE
    assert artifact.headline is None
    assert artifact.date is None
    assert artifact.source is None
    assert artifact.location is None
    assert artifact.text is None
    assert artifact.tags is None


def test_newspaper_artifact_with_fields():
    artifact = NewspaperArtifact(
        thread_id="thread-2",
        headline="Test Headline",
        date="2025-01-01",
        source="Test Source",
        location="Test Location",
        text="Article body text",
        tags=["local", "breaking"],
    )
    assert artifact.headline == "Test Headline"
    assert artifact.date == "2025-01-01"
    assert artifact.source == "Test Source"
    assert artifact.location == "Test Location"
    assert artifact.text == "Article body text"
    assert artifact.tags == ["local", "breaking"]
