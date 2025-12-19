from app.structures.artifacts.audio_artifact_file import AudioChainArtifactFile
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType


def test_defaults():
    """Default attribute values are set on construction."""
    artifact = AudioChainArtifactFile(
        base_path="/",
        chain_artifact_file_type=ChainArtifactFileType.AUDIO,
        audio_bytes=b"test_audio_data",
    )
    assert getattr(artifact, "sample_rate") == 44100
    assert getattr(artifact, "duration") == 1.0
    assert getattr(artifact, "channels") == 2


def test_custom_initialization():
    """Constructor accepts and applies custom values."""
    artifact = AudioChainArtifactFile(
        base_path="/path",
        chain_artifact_file_type=ChainArtifactFileType.AUDIO,
        sample_rate=48000,
        duration=2.5,
        channels=1,
        audio_bytes=b"test_audio_data",
    )
    assert artifact.base_path == "/path"
    assert artifact.chain_artifact_file_type == ChainArtifactFileType.AUDIO
    assert artifact.sample_rate == 48000
    assert artifact.duration == 2.5
    assert artifact.channels == 1


def test_attribute_mutation():
    """Attributes can be updated after construction."""
    artifact = AudioChainArtifactFile(
        base_path="/",
        chain_artifact_file_type=ChainArtifactFileType.AUDIO,
        audio_bytes=b"test_audio_data",
    )
    artifact.sample_rate = 22050
    artifact.duration = 0.75
    artifact.channels = 1
    assert artifact.sample_rate == 22050
    assert artifact.duration == 0.75
    assert artifact.channels == 1


def test_for_prompt_returns_string_and_contains_file_path_and_duration():
    artifact = AudioChainArtifactFile(
        base_path="/media",
        chain_artifact_file_type=ChainArtifactFileType.AUDIO,
        audio_bytes=b"test_audio_data",
    )
    output = artifact.for_prompt()
    assert isinstance(output, str)
    assert str(artifact.file_path) in output
    assert str(artifact.duration) in output


def test_for_prompt_includes_custom_path_and_duration():
    artifact = AudioChainArtifactFile(
        base_path="/custom/path",
        chain_artifact_file_type=ChainArtifactFileType.AUDIO,
        sample_rate=48000,
        duration=2.5,
        channels=1,
        audio_bytes=b"\x00\x01",
    )
    output = artifact.for_prompt()
    assert str(artifact.file_path) in output
    assert str(artifact.duration) in output
    assert "/custom/path" in output


def test_for_prompt_handles_empty_audio_bytes_and_returns_string():
    artifact = AudioChainArtifactFile(
        base_path="/",
        chain_artifact_file_type=ChainArtifactFileType.AUDIO,
        audio_bytes=b"",
    )
    output = artifact.for_prompt()
    assert isinstance(output, str)
    assert str(artifact.duration) in output or str(artifact.sample_rate) in output
