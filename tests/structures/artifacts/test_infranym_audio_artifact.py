from app.structures.artifacts.infranym_audio_artifact import InfranymAudioArtifact

# ToDo: Expand tests when more fields are added to InfranymAudioArtifact


def test_init():
    artifact = InfranymAudioArtifact(thread_id="test")
    assert artifact.thread_id == "test"
