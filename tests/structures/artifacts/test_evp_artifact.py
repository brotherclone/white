from app.structures.artifacts.audio_artifact_file import AudioChainArtifactFile
from app.structures.artifacts.evp_artifact import EVPArtifact
from app.structures.artifacts.text_artifact_file import TextChainArtifactFile
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType


def test_evp_artifact():
    """Test EVPArtifact creation and attributes"""
    artifact = EVPArtifact(thread_id="test-thread-123", chain_artifact_type="evp")
    assert artifact.thread_id == "test-thread-123"
    assert artifact.chain_artifact_type == "evp"
    assert artifact.audio_segments is None
    assert artifact.transcript is None
    assert artifact.audio_mosiac is None
    assert artifact.noise_blended_audio is None


def test_evp_artifact_with_audio_segments():
    """Test EVPArtifact with audio segments"""
    audio_file = AudioChainArtifactFile(
        base_path="/tmp",
        chain_artifact_file_type=ChainArtifactFileType.AUDIO,
        thread_id="test-thread",
    )

    artifact = EVPArtifact(thread_id="test-thread-123", audio_segments=[audio_file])
    assert len(artifact.audio_segments) == 1
    assert isinstance(artifact.audio_segments[0], AudioChainArtifactFile)


def test_evp_artifact_with_transcript():
    """Test EVPArtifact with transcript"""
    transcript = TextChainArtifactFile(
        base_path="/tmp",
        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
        text_content="Test transcript content",
    )

    artifact = EVPArtifact(thread_id="test-thread-123", transcript=transcript)
    assert artifact.transcript is not None
    assert artifact.transcript.text_content == "Test transcript content"
