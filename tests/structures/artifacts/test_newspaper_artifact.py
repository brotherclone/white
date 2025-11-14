from app.structures.artifacts.newspaper_artifact import NewspaperArtifact
from app.structures.artifacts.text_artifact_file import TextChainArtifactFile
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType


def test_newspaper_artifact_defaults():
    artifact = NewspaperArtifact(thread_id="thread-1")
    assert artifact.thread_id == "thread-1"
    assert artifact.chain_artifact_type == "newspaper_article"
    assert artifact.headline is None
    assert artifact.date is None
    assert artifact.source is None
    assert artifact.location is None
    assert artifact.text is None
    assert artifact.tags is None
    assert artifact.page is None


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


def test_newspaper_artifact_page_from_list_of_dicts():
    page_dict = {
        "base_path": "/tmp",
        "chain_artifact_file_type": ChainArtifactFileType.MARKDOWN,
        "text_content": "Page text content",
    }
    artifact = NewspaperArtifact(thread_id="thread-3", page=[page_dict])
    assert isinstance(artifact.page, TextChainArtifactFile)
    assert artifact.page.text_content == "Page text content"
    assert artifact.page.base_path == "/tmp"


def test_newspaper_artifact_page_from_list_of_model():
    page_model = TextChainArtifactFile(
        base_path="/tmp",
        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
        text_content="Another page text",
    )
    artifact = NewspaperArtifact(thread_id="thread-4", page=[page_model])
    assert isinstance(artifact.page, TextChainArtifactFile)
    assert artifact.page.text_content == "Another page text"


def test_get_text_content_contains_fields():
    artifact = NewspaperArtifact(
        thread_id="thread-5",
        headline="Headline Example",
        date="2025-02-02",
        source="Example Source",
        location="Example Location",
        text="Full article text",
    )
    text = artifact.get_text_content()
    assert "Headline Example" in text
    assert "2025-02-02" in text
    assert "Example Source" in text
    assert "Example Location" in text
    assert "Full article text" in text
