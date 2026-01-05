import os
from PIL import Image

from app.structures.artifacts.infranym_text_render_artifact import (
    InfranymTextRenderArtifact,
)
from app.structures.enums.image_text_style import ImageTextStyle


class ConcreteInfranymTextRenderArtifact(InfranymTextRenderArtifact):
    """Concrete implementation for testing"""

    def save_file(self):
        return self.encode()


def test_init_basic():
    """Test basic initialization with required fields."""
    artifact = ConcreteInfranymTextRenderArtifact(
        thread_id="test", secret_word="TEMPORAL"
    )
    assert artifact.thread_id == "test"
    assert artifact.secret_word == "TEMPORAL"
    assert artifact.image_text_style == ImageTextStyle.DEFAULT
    assert artifact.size == (400, 200)


def test_init_with_custom_style():
    """Test initialization with custom style."""
    artifact = ConcreteInfranymTextRenderArtifact(
        thread_id="test", secret_word="GLITCH", image_text_style=ImageTextStyle.GLITCH
    )
    assert artifact.image_text_style == ImageTextStyle.GLITCH


def test_init_with_custom_size():
    """Test initialization with custom size."""
    artifact = ConcreteInfranymTextRenderArtifact(
        thread_id="test", secret_word="SMALL", size=(200, 100)
    )
    assert artifact.size == (200, 100)


def test_all_text_styles():
    """Test creating artifact with each style."""
    for style in ImageTextStyle:
        artifact = ConcreteInfranymTextRenderArtifact(
            thread_id="test", secret_word="TEST", image_text_style=style
        )
        assert artifact.image_text_style == style


def test_create_text_image_default():
    """Test creating text image with default style."""
    artifact = ConcreteInfranymTextRenderArtifact(
        thread_id="test", secret_word="DEFAULT"
    )
    img = artifact.create_text_image()

    assert isinstance(img, Image.Image)
    assert img.size == (400, 200)
    assert img.mode == "RGB"


def test_create_text_image_clean():
    """Test creating text image with clean style."""
    artifact = ConcreteInfranymTextRenderArtifact(
        thread_id="test", secret_word="CLEAN", image_text_style=ImageTextStyle.CLEAN
    )
    img = artifact.create_text_image(style=ImageTextStyle.CLEAN)

    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_create_text_image_glitch():
    """Test creating text image with glitch style."""
    artifact = ConcreteInfranymTextRenderArtifact(
        thread_id="test", secret_word="GLITCH", image_text_style=ImageTextStyle.GLITCH
    )
    img = artifact.create_text_image(style=ImageTextStyle.GLITCH)

    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_create_text_image_static():
    """Test creating text image with static/noise style."""
    artifact = ConcreteInfranymTextRenderArtifact(
        thread_id="test", secret_word="STATIC", image_text_style=ImageTextStyle.STATIC
    )
    img = artifact.create_text_image(style=ImageTextStyle.STATIC)

    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_create_text_image_custom_size():
    """Test creating text image with custom size."""
    artifact = ConcreteInfranymTextRenderArtifact(
        thread_id="test", secret_word="CUSTOM"
    )
    img = artifact.create_text_image(size=(800, 400))

    assert img.size == (800, 400)


def test_for_prompt():
    """Test for_prompt() method."""
    artifact = ConcreteInfranymTextRenderArtifact(
        thread_id="test", secret_word="PROMPT", image_text_style=ImageTextStyle.CLEAN
    )
    prompt = artifact.for_prompt()

    assert "Text Render:" in prompt
    assert "PROMPT" in prompt
    assert "clean" in prompt


def test_for_prompt_all_styles():
    """Test for_prompt() includes correct style."""
    for style in ImageTextStyle:
        artifact = ConcreteInfranymTextRenderArtifact(
            thread_id="test", secret_word="WORD", image_text_style=style
        )
        prompt = artifact.for_prompt()
        assert style.value in prompt


def test_flatten():
    """Test flatten() method."""
    artifact = ConcreteInfranymTextRenderArtifact(
        thread_id="flatten_test",
        secret_word="FLATTEN",
        image_text_style=ImageTextStyle.GLITCH,
        size=(300, 150),
    )
    flat = artifact.flatten()

    assert flat["secret_word"] == "FLATTEN"
    assert flat["image_text_style"] == ImageTextStyle.GLITCH.value
    assert flat["size"] == (300, 150)


def test_flatten_default_values():
    """Test flatten() with default values."""
    artifact = ConcreteInfranymTextRenderArtifact(
        thread_id="test", secret_word="DEFAULT"
    )
    flat = artifact.flatten()

    assert flat["secret_word"] == "DEFAULT"
    assert flat["image_text_style"] == ImageTextStyle.DEFAULT.value
    assert flat["size"] == (400, 200)


def test_encode_creates_image(tmp_path):
    """Test encode() creates an image file."""
    artifact = ConcreteInfranymTextRenderArtifact(
        thread_id="encode_test",
        base_path=str(tmp_path),
        secret_word="ENCODE",
        chain_artifact_file_type="png",
        artifact_name="test_render",
    )

    output_path = artifact.encode()

    assert os.path.exists(output_path)
    assert output_path.endswith(".png")

    # Verify it's a valid image
    img = Image.open(output_path)
    assert isinstance(img, Image.Image)


def test_encode_different_styles(tmp_path):
    """Test encoding with different styles produces valid images."""
    for style in ImageTextStyle:
        artifact = ConcreteInfranymTextRenderArtifact(
            thread_id=f"style_{style.value}",
            base_path=str(tmp_path),
            secret_word=style.value.upper(),
            image_text_style=style,
            chain_artifact_file_type="png",
            artifact_name=f"render_{style.value}",
        )

        output_path = artifact.encode()
        assert os.path.exists(output_path)

        # Verify valid image
        img = Image.open(output_path)
        assert img.mode == "RGB"


def test_different_secret_words():
    """Test with various secret words."""
    words = ["TEMPORAL", "LIMINAL", "THRESHOLD", "ECHO", "VOID"]

    for word in words:
        artifact = ConcreteInfranymTextRenderArtifact(
            thread_id="test", secret_word=word
        )
        assert artifact.secret_word == word

        img = artifact.create_text_image()
        assert isinstance(img, Image.Image)


def test_small_size():
    """Test with small image size."""
    artifact = ConcreteInfranymTextRenderArtifact(
        thread_id="test", secret_word="TINY", size=(100, 50)
    )
    img = artifact.create_text_image(size=(100, 50))

    assert img.size == (100, 50)


def test_large_size():
    """Test with large image size."""
    artifact = ConcreteInfranymTextRenderArtifact(
        thread_id="test", secret_word="BIG", size=(1000, 500)
    )
    img = artifact.create_text_image(size=(1000, 500))

    assert img.size == (1000, 500)


def test_long_secret_word():
    """Test with a long secret word."""
    artifact = ConcreteInfranymTextRenderArtifact(
        thread_id="test", secret_word="VERYLONGSECRETWORD"
    )
    img = artifact.create_text_image()

    assert isinstance(img, Image.Image)


def test_single_character_word():
    """Test with single character."""
    artifact = ConcreteInfranymTextRenderArtifact(thread_id="test", secret_word="X")
    img = artifact.create_text_image()

    assert isinstance(img, Image.Image)


def test_image_has_content():
    """Test that created images have non-black content (text was rendered)."""
    artifact = ConcreteInfranymTextRenderArtifact(
        thread_id="test", secret_word="TEST", image_text_style=ImageTextStyle.CLEAN
    )
    img = artifact.create_text_image()

    # Convert to array and check if there's any non-black pixels
    import numpy as np

    arr = np.array(img)

    # For clean/default styles on black background, should have white pixels
    # (This is a simple sanity check that text was rendered)
    has_non_black = (arr > 0).any()
    assert (
        has_non_black
    ), "Image appears to be entirely black - text may not have rendered"


def test_static_style_has_noise():
    """Test that static style has noise background."""
    artifact = ConcreteInfranymTextRenderArtifact(
        thread_id="test", secret_word="NOISE", image_text_style=ImageTextStyle.STATIC
    )
    img = artifact.create_text_image(style=ImageTextStyle.STATIC)

    import numpy as np

    arr = np.array(img)

    # Static style should have varied pixel values due to noise
    # Check that we have a range of values, not just black
    unique_values = len(np.unique(arr))
    assert unique_values > 10, "Static style should have noise variation"
