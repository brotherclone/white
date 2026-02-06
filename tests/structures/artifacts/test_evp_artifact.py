from app.structures.artifacts.audio_artifact_file import AudioChainArtifactFile
from app.structures.artifacts.evp_artifact import EVPArtifact
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType


def test_evp_artifact():
    """Test EVPArtifact creation and attributes"""
    artifact = EVPArtifact(thread_id="mock_thread_00123")
    assert artifact.thread_id == "mock_thread_00123"
    assert artifact.chain_artifact_type == ChainArtifactType.EVP_ARTIFACT
    assert artifact.transcript is None
    assert artifact.audio_mosiac is None


def test_evp_artifact_with_mosaic():
    """Test EVPArtifact with mosaic audio"""
    audio_file = AudioChainArtifactFile(
        base_path="/tmp",
        chain_artifact_file_type=ChainArtifactFileType.AUDIO,
        thread_id="test-thread",
        audio_bytes=b"fake audio data",
    )

    artifact = EVPArtifact(thread_id="mock_thread_001", audio_mosiac=audio_file)
    assert isinstance(artifact.audio_mosiac, AudioChainArtifactFile)


def test_evp_artifact_with_transcript():
    """Test EVPArtifact with transcript"""
    artifact = EVPArtifact(
        thread_id="mock_thread_001", transcript="Test transcript content"
    )
    assert artifact.transcript is not None
    assert artifact.transcript == "Test transcript content"


def test_evp_artifact_ignores_legacy_fields():
    """Test that EVPArtifact gracefully ignores legacy audio_segments and noise_blended_audio"""
    artifact = EVPArtifact(
        thread_id="mock_thread_001",
        transcript="test",
        audio_segments=["fake_segment_path.wav"],
        noise_blended_audio="fake_blended_path.wav",
    )
    assert artifact.transcript == "test"
    assert not hasattr(artifact, "audio_segments")
    assert not hasattr(artifact, "noise_blended_audio")


def test_evp_artifact_only_mosaic_saved(tmp_path, monkeypatch):
    """Test that save_file only saves the mosaic, not segments or blended"""
    thread_id = "mock_thread_001"
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", str(tmp_path))
    artifact = EVPArtifact(
        thread_id=thread_id,
        transcript="test transcript",
        base_path=str(tmp_path),
    )
    artifact.save_file()
    yml_dir = tmp_path / thread_id / "yml"
    assert yml_dir.exists()
    yml_files = list(yml_dir.glob("*.yml"))
    assert len(yml_files) == 1
    import yaml

    with open(yml_files[0], "r") as f:
        data = yaml.safe_load(f)
    assert "audio_segments" not in data
    assert "noise_blended_audio" not in data
    assert "transcript" in data
    assert "audio_mosiac" in data
