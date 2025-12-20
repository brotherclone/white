from app.structures.artifacts.audio_artifact_file import AudioChainArtifactFile
from app.structures.artifacts.evp_artifact import EVPArtifact
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType


def test_evp_artifact():
    """Test EVPArtifact creation and attributes"""
    artifact = EVPArtifact(thread_id="test-thread-123")
    assert artifact.thread_id == "test-thread-123"
    assert artifact.chain_artifact_type == ChainArtifactType.EVP_ARTIFACT
    assert artifact.audio_segments == []
    assert artifact.transcript is None
    assert artifact.audio_mosiac is None
    assert artifact.noise_blended_audio is None


def test_evp_artifact_with_audio_segments():
    """Test EVPArtifact with audio segments"""
    audio_file = AudioChainArtifactFile(
        base_path="/tmp",
        chain_artifact_file_type=ChainArtifactFileType.AUDIO,
        thread_id="test-thread",
        audio_bytes=b"fake audio data",
    )

    artifact = EVPArtifact(thread_id="test-thread-123", audio_segments=[audio_file])
    assert len(artifact.audio_segments) == 1
    assert isinstance(artifact.audio_segments[0], AudioChainArtifactFile)


def test_evp_artifact_with_transcript():
    """Test EVPArtifact with transcript"""
    artifact = EVPArtifact(
        thread_id="test-thread-123", transcript="Test transcript content"
    )
    assert artifact.transcript is not None
    assert artifact.transcript == "Test transcript content"
