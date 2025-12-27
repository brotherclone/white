from app.structures.artifacts.html_artifact_file import HtmlChainArtifactFile


def test_html_artifact_file():
    """Test basic HtmlChainArtifactFile instantiation"""
    artifact = HtmlChainArtifactFile(thread_id="test", image_path="test.png")
    assert artifact.thread_id == "test"
    assert artifact.image_path == "test.png"
