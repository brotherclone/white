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
