from app.structures.artifacts.infranym_audio_artifact import InfranymAudioArtifact


def test_init():
    artifact = InfranymAudioArtifact(thread_id="test")
    assert artifact.thread_id == "test"
