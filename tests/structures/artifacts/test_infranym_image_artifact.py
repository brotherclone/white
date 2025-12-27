from app.structures.artifacts.infranym_image_artifact import InfranymImageArtifact


def test_init():
    artifact = InfranymImageArtifact(thread_id="test")
    assert artifact.thread_id == "test"
