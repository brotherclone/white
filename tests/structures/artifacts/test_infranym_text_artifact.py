from app.structures.artifacts.infranym_text_artifact import InfranymTextArtifact

# ToDo: Expand tests when more fields are added to InfranymAudioArtifact


def test_init():
    artifact = InfranymTextArtifact(thread_id="test")
    assert artifact.thread_id == "test"
