import pytest

from pydantic import ValidationError

from app.structures.artifacts.newspaper_artifact import NewspaperArtifact
from app.structures.enums.chain_artifact_type import ChainArtifactType
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType


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
    # defaults from base artifact
    assert artifact.chain_artifact_file_type == ChainArtifactFileType.YML
    assert artifact.file_name is not None
    assert artifact.file_path is not None


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


def test_get_text_content_empty_and_long():
    a = NewspaperArtifact(thread_id="t3", text=None)
    assert a.get_text_content() == ""

    long_text = "x" * 5000
    a2 = NewspaperArtifact(thread_id="t4", text=long_text)
    assert a2.get_text_content() == long_text


def test_to_markdown_contains_key_sections():
    artifact = NewspaperArtifact(
        thread_id="t-md",
        headline="Big News",
        date="2025-02-02",
        source="The Paper",
        location="Hometown",
        text="Important article body.",
        tags=["one", "two"],
    )
    md = artifact.to_markdown()
    assert md.startswith("# Big News")
    assert "**Source:** The Paper" in md
    assert "**Date:** 2025-02-02" in md
    assert "**Location:** Hometown" in md
    assert "Important article body." in md
    assert "**Tags:** one, two" in md
    assert "## Metadata" in md
    assert f"- **Thread ID:** {artifact.thread_id}" in md


def test_for_prompt_with_and_without_tags():
    a_no_tags = NewspaperArtifact(thread_id="t-p1", headline="H1", text="T1", tags=None)
    p_no = a_no_tags.for_prompt()
    assert "H1" in p_no
    assert "T1" in p_no
    assert "[" not in p_no  # no tags bracket

    a_with_tags = NewspaperArtifact(
        thread_id="t-p2", headline="H2", text="T2", tags=["a", "b"]
    )
    p_with = a_with_tags.for_prompt()
    assert "H2" in p_with
    assert "T2" in p_with
    assert "[a, b]" in p_with


def test_flatten_and_roundtrip_model_dump():
    artifact = NewspaperArtifact(
        thread_id="t-round",
        headline="RT",
        text="Roundtrip",
        tags=["x"],
    )
    flat = artifact.flatten()
    # required keys present
    for key in [
        "thread_id",
        "chain_artifact_file_type",
        "file_name",
        "file_path",
        "chain_artifact_type",
        "headline",
        "text",
        "tags",
    ]:
        assert key in flat

    assert flat["chain_artifact_type"] == ChainArtifactType.NEWSPAPER_ARTICLE.value

    # roundtrip using model_dump / constructor
    dumped = artifact.model_dump(mode="python")
    new = NewspaperArtifact(**dumped)
    assert new.headline == artifact.headline
    assert new.text == artifact.text
    assert new.tags == artifact.tags
    assert new.thread_id == artifact.thread_id


def test_artifact_name_sanitization_and_file_name_immutable():
    bad_headline = "A / Bad: Headline"
    a = NewspaperArtifact(thread_id="t-s", headline=bad_headline)
    # artifact_name should be sanitized (no path separators)
    assert "/" not in a.artifact_name
    assert ":" not in a.artifact_name
    original_file_name = a.file_name
    # mutating headline should not change existing file_name
    a.headline = "Completely Different"
    assert a.file_name == original_file_name


def test_tags_type_validation():
    # tags should be list[str] or None; passing a string should raise ValidationError
    with pytest.raises(ValidationError):
        NewspaperArtifact(thread_id="t-val", tags="not-a-list")


def test_to_markdown_with_minimal_fields():
    """Test to_markdown with only required fields."""
    artifact = NewspaperArtifact(thread_id="t-min")
    md = artifact.to_markdown()
    # Should still produce valid markdown with metadata section
    assert "## Metadata" in md
    assert "Thread ID" in md
    assert "t-min" in md


def test_to_markdown_all_none_optional_fields():
    """Test to_markdown when all optional fields are None."""
    artifact = NewspaperArtifact(
        thread_id="t-none",
        headline=None,
        date=None,
        source=None,
        location=None,
        text=None,
        tags=None,
    )
    md = artifact.to_markdown()
    # Should produce valid markdown without errors
    assert isinstance(md, str)
    assert "## Metadata" in md


def test_for_prompt_with_only_headline():
    """Test for_prompt with only headline."""
    artifact = NewspaperArtifact(thread_id="t-h", headline="Solo Headline")
    prompt = artifact.for_prompt()
    assert "Solo Headline" in prompt


def test_for_prompt_with_all_metadata():
    """Test for_prompt with all metadata fields."""
    artifact = NewspaperArtifact(
        thread_id="t-all",
        headline="Full Story",
        source="Daily Times",
        date="2025-12-27",
        location="New York, NY",
        text="Complete article text here.",
        tags=["breaking", "local", "politics"],
    )
    prompt = artifact.for_prompt()
    assert "Full Story" in prompt
    assert "Daily Times" in prompt
    assert "2025-12-27" in prompt
    assert "New York, NY" in prompt
    assert "Complete article text here." in prompt
    assert "breaking" in prompt
    assert "local" in prompt
    assert "politics" in prompt


def test_for_prompt_metadata_separator():
    """Test that metadata is properly separated with pipes."""
    artifact = NewspaperArtifact(
        thread_id="t-sep", source="Source A", date="2025-01-01", location="Place B"
    )
    prompt = artifact.for_prompt()
    assert "Source A | 2025-01-01 | Place B" in prompt


def test_artifact_name_from_headline():
    """Test that artifact_name is auto-generated from headline."""
    artifact = NewspaperArtifact(
        thread_id="t-name", headline="Breaking: Major Event Occurs!"
    )
    # Should have artifact_name derived from headline
    assert artifact.artifact_name is not None
    assert len(artifact.artifact_name) > 0
    # Should be sanitized (no special chars like colons or exclamation marks)
    assert ":" not in artifact.artifact_name
    assert "!" not in artifact.artifact_name


def test_empty_tags_list():
    """Test with empty tags list."""
    artifact = NewspaperArtifact(thread_id="t-empty-tags", tags=[])
    assert artifact.tags == []
    md = artifact.to_markdown()
    # Empty tags should not appear in markdown
    assert "**Tags:**" not in md


def test_get_text_content_with_text():
    """Test get_text_content returns text correctly."""
    text = "This is the full article text with multiple sentences."
    artifact = NewspaperArtifact(thread_id="t-text", text=text)
    assert artifact.get_text_content() == text


def test_chain_artifact_type_immutable():
    """Test that chain_artifact_type is always NEWSPAPER_ARTICLE."""
    artifact = NewspaperArtifact(thread_id="t-type")
    assert artifact.chain_artifact_type == ChainArtifactType.NEWSPAPER_ARTICLE
    # Verify it's in the flatten output correctly
    flat = artifact.flatten()
    assert flat["chain_artifact_type"] == ChainArtifactType.NEWSPAPER_ARTICLE.value
