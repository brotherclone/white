import logging
import numpy as np
import os

from typing import Tuple
from dotenv import load_dotenv
from abc import ABC
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.concepts.rainbow_table_color import get_rainbow_table_color
from app.structures.enums.image_text_style import ImageTextStyle

load_dotenv()


class InfranymTextRenderArtifact(ChainArtifact, ABC):
    """
    Layer 2 Source: Renders secret word as a styled image.

    This artifact creates the visual text image that will be hidden
    via LSB steganography in the encoded puzzle image. The rendered
    text serves as both the hidden layer and the decryption key.
    """

    secret_word: str = Field(..., description="Secret word to render as image")
    image_text_style: ImageTextStyle = Field(
        default=ImageTextStyle.DEFAULT, description="Visual style for text rendering"
    )
    size: Tuple[int, int] = Field(
        default=(400, 200),
        description="Image dimensions (width, height) - smaller = fits in smaller carriers",
    )

    def __init__(self, **data):
        super().__init__(**data)

    def encode(self) -> str:
        """
        Render the secret word as a styled image.

        Returns:
            Path to the saved text render image
        """
        text_img = self.create_text_image(size=self.size, style=self.image_text_style)

        # Get path and create directories
        output_path = self.get_artifact_path(with_file_name=True, create_dirs=True)
        text_img.save(output_path, format="PNG")

        logging.info(f"💜 Text render saved: {output_path}")
        return output_path

    def create_text_image(
        self,
        size: Tuple[int, int] = (400, 200),
        style: ImageTextStyle = ImageTextStyle.DEFAULT,
    ) -> Image.Image:
        """
        Render secret word with a specified style.

        Args:
            size: Image dimensions
            style: Visual style enum (uses enum comparison, not strings)

        Returns:
            PIL Image with rendered text
        """
        # Get rainbow colors for styling
        w = get_rainbow_table_color("A")  # White
        r = get_rainbow_table_color("R")  # Red
        g = get_rainbow_table_color("G")  # Green
        b = get_rainbow_table_color("B")  # Blue

        img = Image.new("RGB", size, color="black")
        draw = ImageDraw.Draw(img)

        # Use enum comparison, not string comparison
        if style == ImageTextStyle.CLEAN:
            try:
                font = ImageFont.truetype(
                    f"{os.getenv('LOCAL_FONT_PATH')}/HelveticaNeue.ttc", 120
                )
            except (EnvironmentError, OSError) as e:
                logging.error(f"Error loading font: {e}")
                font = ImageFont.load_default()
            fill = w.hex_value

        elif style == ImageTextStyle.GLITCH:
            try:
                font = ImageFont.truetype(
                    f"{os.getenv('LOCAL_FONT_PATH')}/Courier.ttc", 100
                )
            except (EnvironmentError, OSError) as e:
                logging.error(f"Error loading font: {e}")
                font = ImageFont.load_default()

            # Multi-offset RGB separation for glitch effect
            for offset in [(0, 0), (2, 2), (-2, -2), (4, 0)]:
                bbox = draw.textbbox((0, 0), self.secret_word, font=font)
                x = (size[0] - (bbox[2] - bbox[0])) / 2 + offset[0]
                y = (size[1] - (bbox[3] - bbox[1])) / 2 + offset[1]
                draw.text(
                    (x, y),
                    self.secret_word,
                    fill=(r.hex_value, g.hex_value, b.hex_value, w.hex_value)[
                        offset[0] % 4
                    ],
                    font=font,
                )
            return img

        elif style == ImageTextStyle.STATIC:
            # Noise background for EVP aesthetic
            noise = np.random.randint(0, 50, (size[1], size[0], 3), dtype="uint8")
            img = Image.fromarray(noise)
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype(
                    f"{os.getenv('LOCAL_FONT_PATH')}/Monaco.ttf", 80
                )
            except (EnvironmentError, OSError) as e:
                logging.error(f"Error loading font: {e}")
                font = ImageFont.load_default()
            fill = w.hex_value

        else:  # DEFAULT or any other value
            try:
                font = ImageFont.truetype(
                    f"{os.getenv('LOCAL_FONT_PATH')}/Helvetica.ttc", 120
                )
            except (EnvironmentError, OSError) as e:
                logging.error(f"Error loading font: {e}")
                font = ImageFont.load_default()
            fill = w.hex_value

        # Center the text (unless glitch already drew it)
        if style != ImageTextStyle.GLITCH:
            bbox = draw.textbbox((0, 0), self.secret_word, font=font)
            x = (size[0] - (bbox[2] - bbox[0])) / 2
            y = (size[1] - (bbox[3] - bbox[1])) / 2
            draw.text((x, y), self.secret_word, fill=fill, font=font)

        return img

    def flatten(self):
        """Serialize for state persistence"""
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}
        return {
            **parent_data,
            "secret_word": self.secret_word,
            "image_text_style": self.image_text_style.value,
            "size": self.size,
        }

    def for_prompt(self) -> str:
        """Format for LLM context"""
        return (
            f"Text Render: '{self.secret_word}' (style: {self.image_text_style.value})"
        )

    def save_file(self):
        """Legacy method - use encode() instead"""
        return self.encode()


if __name__ == "__main__":
    # Test text render with recommended default size
    text_render = InfranymTextRenderArtifact(
        base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
        thread_id="mock_thread_001",
        chain_artifact_file_type="png",
        chain_artifact_type="infranym_text_render",
        rainbow_color_mnemonic_character_value="I",
        artifact_name="word_TEMPORAL",
        secret_word="TEMPORAL",
        image_text_style=ImageTextStyle.STATIC,
        # size defaults to (400, 200) - recommended for most carriers
    )

    path = text_render.encode()
    print(f"✅ Text render created: {path}")
    print(f"📄 Flattened: {text_render.flatten()}")
    print(f"💬 For prompt: {text_render.for_prompt()}")
