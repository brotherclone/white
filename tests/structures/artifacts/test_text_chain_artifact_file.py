from app.structures.artifacts.text_chain_artifact_file import TextChainArtifactFile
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType


def test_text_chain_artifact_file():
    """Test TextChainArtifactFile creation with text content"""
    artifact = TextChainArtifactFile(
        base_path="/tmp",
        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
        text_content="Hello, world!",
    )
    assert artifact.base_path == "/tmp"
    assert artifact.chain_artifact_file_type == ChainArtifactFileType.MARKDOWN
    assert artifact.text_content == "Hello, world!"
    assert artifact.artifact_id is not None


def test_text_chain_artifact_file_empty_content():
    """Test TextChainArtifactFile with no text content"""
    artifact = TextChainArtifactFile(
        base_path="/tmp/text", chain_artifact_file_type=ChainArtifactFileType.MARKDOWN
    )
    assert artifact.text_content is None


def test_text_chain_artifact_file_with_thread_id():
    """Test TextChainArtifactFile with thread_id"""
    artifact = TextChainArtifactFile(
        base_path="/tmp",
        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
        text_content="Test content",
        thread_id="test-thread-789",
    )
    assert artifact.thread_id == "test-thread-789"
