from pathlib import Path
from app.structures.artifacts.image_artifact_file import ImageChainArtifactFile


class ConcreteImageChainArtifactFile(ImageChainArtifactFile):
    """Concrete implementation for testing"""

    def flatten(self):
        parent_data = super().flatten() if hasattr(super(), "flatten") else {}
        return {
            **parent_data,
            "height": self.height,
            "width": self.width,
            "aspect_ratio": self.aspect_ratio,
        }


def test_init():
    image = ConcreteImageChainArtifactFile(
        thread_id="test-thread", file_path=Path("/tmp/test.png"), height=100, width=200
    )
    assert image.thread_id == "test-thread"
    assert image.height == 100
    assert image.width == 200
    assert image.aspect_ratio == 1.0


def test_for_prompt_returns_string_and_contains_artifact_name():
    image = ConcreteImageChainArtifactFile(
        thread_id="test-thread", height=10, width=20, artifact_name="portrait"
    )
    out = image.for_prompt()
    assert isinstance(out, str)
    assert "portrait" in out
    assert out.startswith("Image of")


def test_for_prompt_preserves_explicit_file_path_and_returns_string():
    explicit_path = Path("/tmp/test.png")
    image = ConcreteImageChainArtifactFile(
        thread_id="test-thread", file_path=explicit_path, height=50, width=100
    )
    assert image.file_path == explicit_path
    out = image.for_prompt()
    assert isinstance(out, str)
