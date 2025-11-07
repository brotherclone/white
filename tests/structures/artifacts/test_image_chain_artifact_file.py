from app.structures.artifacts.image_chain_artifact_file import ImageChainArtifactFile
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType


def test_image_chain_artifact_file():
    """Test ImageChainArtifactFile creation"""
    artifact = ImageChainArtifactFile(
        base_path="/tmp", chain_artifact_file_type=ChainArtifactFileType.PNG
    )
    assert artifact.base_path == "/tmp"
    assert artifact.chain_artifact_file_type == ChainArtifactFileType.PNG
    assert artifact.artifact_id is not None


def test_image_chain_artifact_file_with_thread_id():
    """Test ImageChainArtifactFile with thread_id"""
    artifact = ImageChainArtifactFile(
        base_path="/tmp/images",
        chain_artifact_file_type=ChainArtifactFileType.PNG,
        thread_id="test-thread-456",
    )
    assert artifact.thread_id == "test-thread-456"
    assert artifact.base_path == "/tmp/images"
