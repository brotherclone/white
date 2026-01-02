from app.structures.artifacts.infranym_midi_artifact import InfranymMidiArtifact

# ToDo: Expand tests when more fields are added to InfranymAudioArtifact


def test_init():
    artifact = InfranymMidiArtifact(thread_id="test")
    assert artifact.thread_id == "test"
