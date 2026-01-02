import logging
import os
import numpy as np
import base64

from dotenv import load_dotenv
from typing import Optional
from abc import ABC
from PIL import Image, PngImagePlugin
from pydantic import Field
from stegano import lsb

from app.structures.artifacts.base_artifact import ChainArtifact

load_dotenv()


class InfranymEncodedImageArtifact(ChainArtifact, ABC):
    """
    Complete Infranym Puzzle: All three layers embedded in carrier image.

    Layer 1: PNG metadata (surface clue, discoverable with exiftool)
    Layer 2: Text render image hidden via LSB steganography
    Layer 3: Solution encrypted via spread spectrum (keyed by Layer 2 word)

    The puzzle requires progressive revelation:
    1. Find metadata clue (easy)
    2. Extract LSB to reveal a text image (medium)
    3. Use revealed word as a key to decrypt a solution (hard)
    """

    carrier_image_path: str = Field(..., description="Path to carrier image")
    text_render_path: str = Field(
        ..., description="Path to rendered text image (Layer 2 source)"
    )
    surface_clue: str = Field(..., description="Surface puzzle clue (Layer 1)")
    solution: str = Field(..., description="Final solution text (Layer 3)")
    secret_word: Optional[str] = Field(
        default=None,
        description="Secret word (extracted from text_render if not provided)",
    )

    def __init__(self, **data):
        super().__init__(**data)

    @staticmethod
    def calculate_required_carrier_size(
        text_render_path: str, solution_length: int
    ) -> tuple[int, int]:
        """
        Calculate the minimum carrier image size needed to hide the data.

        Args:
            text_render_path: Path to text render image
            solution_length: Length of solution text

        Returns:
            Tuple of (min_width, min_height) required for carrier
        """
        text_img = Image.open(text_render_path)
        width, height = text_img.size

        # Calculate data size
        # Format: "width,height|base64_pixel_data"
        pixel_count = width * height * 3  # RGB
        pixel_data_size = pixel_count * 4 // 3  # base64 expansion
        dimensions_size = len(f"{width},{height}|")
        layer2_size = dimensions_size + pixel_data_size
        layer3_size = solution_length * 8  # 8 bits per character

        # Total data to hide (Layer 2 LSB + Layer 3 spread spectrum)
        total_bits = (layer2_size * 8) + layer3_size

        # LSB can hide 1 bit per pixel, need 3 channels
        required_pixels = total_bits // 3

        # Assume square-ish image, add 20% safety margin
        required_pixels = int(required_pixels * 1.2)
        side = int(np.sqrt(required_pixels)) + 1

        return side, side

    def encode(self) -> str:
        """
        Embed all three layers into carrier image.

        Returns:
            Path to the saved encoded puzzle image
        """
        # Validate carrier size before encoding
        carrier = Image.open(self.carrier_image_path)
        required_size = self.calculate_required_carrier_size(
            self.text_render_path, len(self.solution)
        )

        carrier_pixels = carrier.width * carrier.height
        required_pixels = required_size[0] * required_size[1]

        if carrier_pixels < required_pixels:
            raise ValueError(
                f"Carrier image too small! "
                f"Carrier: {carrier.width}x{carrier.height} ({carrier_pixels:,} pixels), "
                f"Required: {required_size[0]}x{required_size[1]} ({required_pixels:,} pixels). "
                f"Increase carrier size or reduce text render dimensions."
            )

        logging.info(
            f"💜 Carrier size OK: {carrier.width}x{carrier.height} ({carrier_pixels:,} pixels)"
        )

        # LAYER 1: PNG Metadata (surface clue)
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("InfranymSurface", self.surface_clue)
        metadata.add_text("InfranymDepth", "3")
        metadata.add_text("InfranymHint", "Look deeper than metadata")

        # LAYER 2: Load pre-rendered text image and encode as raw pixel data
        text_img = Image.open(self.text_render_path)

        # Convert to raw pixel data (much smaller than full PNG)
        text_array = np.array(text_img)
        height, width = text_array.shape[:2]

        # Create a compact representation: dimensions + raw RGB bytes
        pixel_data = text_array.tobytes()
        compact_data = (
            f"{width},{height}|{base64.b64encode(pixel_data).decode('utf-8')}"
        )

        # Hide compact data in carrier via LSB
        carrier_with_text = lsb.hide(self.carrier_image_path, compact_data)

        # LAYER 3: Spread spectrum using secret word as key
        if not self.secret_word:
            # Try to extract from the text_render_path filename
            self.secret_word = self._extract_word_from_path(self.text_render_path)

        carrier_array = np.array(carrier_with_text)
        final_array = self.spread_spectrum_embed(
            carrier_array, key=self.secret_word, message=self.solution
        )

        # Save final encoded image
        final_img = Image.fromarray(final_array.astype("uint8"))
        output_path = self.get_artifact_path(with_file_name=True, create_dirs=True)
        final_img.save(output_path, format="PNG", pnginfo=metadata)

        logging.info(f"💜 Encoded puzzle saved: {output_path}")
        logging.info(f"   L1: {self.surface_clue[:50]}...")
        logging.info(f"   L2: Text image from {self.text_render_path}")
        logging.info(f"   L3: Solution encrypted with key '{self.secret_word}'")

        return output_path

    def spread_spectrum_embed(
        self, img_array: np.ndarray, key: str, message: str
    ) -> np.ndarray:
        """
        Hide message using secret word as a pseudorandom key.

        Args:
            img_array: Image as numpy array
            key: Secret word used as seed for pseudorandom distribution
            message: Solution text to hide

        Returns:
            Modified image array with an embedded message
        """
        np.random.seed(hash(key) % (2**32))  # Use word as a deterministic seed

        # Convert message to binary
        msg_bits = "".join(format(ord(c), "08b") for c in message)

        # Generate pseudorandom positions for a bit distribution
        flat = img_array.flatten()
        positions = np.random.permutation(len(flat))[: len(msg_bits)]

        # Embed bits at those positions
        for pos, bit in zip(positions, msg_bits):
            flat[pos] = (flat[pos] & 0xFE) | int(bit)

        return flat.reshape(img_array.shape)

    def extract_layer2_text(self, save_revealed: bool = True) -> Optional[Image.Image]:
        """
        Extract the hidden text image from Layer 2.

        Args:
            save_revealed: If True, save the extracted text image

        Returns:
            PIL Image of the revealed text, or None if extraction fails
        """
        encoded_path = self.get_artifact_path(with_file_name=True)
        compact_data = lsb.reveal(encoded_path)

        if not compact_data:
            logging.warning("No LSB data found in image")
            return None

        try:
            # Parse compact format: "width,height|base64_pixel_data"
            dimensions, pixel_b64 = compact_data.split("|")
            width, height = map(int, dimensions.split(","))

            pixel_bytes = base64.b64decode(pixel_b64)
            pixel_array = np.frombuffer(pixel_bytes, dtype=np.uint8)
            pixel_array = pixel_array.reshape((height, width, 3))
            text_img = Image.fromarray(pixel_array, mode="RGB")

            if save_revealed:
                reveal_path = encoded_path.replace(".png", "_LAYER2_REVEALED.png")
                text_img.save(reveal_path)
                logging.info(f"🩵 Layer 2 text image revealed: {reveal_path}")

            return text_img
        except Exception as e:
            logging.error(f"Error extracting Layer 2: {e}")
            return None

    def solve_layer3(self, secret_word_key: Optional[str] = None) -> Optional[str]:
        """
        Decrypt Layer 3 solution using the secret word from Layer 2.

        Args:
            secret_word_key: The word from Layer 2 (uses self.secret_word if not provided)

        Returns:
            Decrypted solution text, or None if decryption fails
        """
        if not secret_word_key:
            secret_word_key = self.secret_word

        if not secret_word_key:
            logging.error("No secret word provided to decrypt Layer 3")
            return None

        try:
            encoded_path = self.get_artifact_path(with_file_name=True)
            img = Image.open(encoded_path)
            img_array = np.array(img)

            # Use same seed as encoding
            np.random.seed(hash(secret_word_key) % (2**32))

            flat = img_array.flatten()
            msg_length = 1000
            positions = np.random.permutation(len(flat))[:msg_length]

            # Extract bits from pseudorandom positions
            bits = "".join(str(flat[pos] & 1) for pos in positions)
            chars = [chr(int(bits[i : i + 8], 2)) for i in range(0, len(bits), 8)]
            message = "".join(chars).split("\x00")[0]  # Stop at null terminator

            logging.info(f" 🩵Layer 3 decrypted with key '{secret_word_key}'")
            return message

        except Exception as e:
            logging.error(f"Error solving Layer 3: {e}")
            return None

    @staticmethod
    def _extract_word_from_path(path: str) -> str:
        """
        Extract secret word from text render artifact filename.

        Expected format: word_SECRETWORD.png or similar
        """
        import os

        filename = os.path.basename(path)
        # Try to extract word after "word_" prefix
        if "word_" in filename:
            word = filename.split("word_")[1].split(".")[0]
            return word.upper()
        return filename.split(".")[0].upper()

    def flatten(self):
        """Serialize for state persistence"""
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}
        return {
            **parent_data,
            "carrier_image_path": self.carrier_image_path,
            "text_render_path": self.text_render_path,
            "surface_clue": self.surface_clue,
            "solution": self.solution,
            "secret_word": self.secret_word,
        }

    def for_prompt(self) -> str:
        """Format for LLM context"""
        return (
            f"Infranym Puzzle: '{self.artifact_name}' (3 layers, keyed by text render)"
        )

    def save_file(self):
        """Legacy method - use encode() instead"""
        return self.encode()


if __name__ == "__main__":
    from app.structures.artifacts.infranym_text_render_artifact import (
        InfranymTextRenderArtifact,
    )
    from app.structures.enums.image_text_style import ImageTextStyle

    # Step 1: Render text with default 400x200 size
    text_render = InfranymTextRenderArtifact(
        base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
        thread_id="test_thread_002",
        chain_artifact_file_type="png",
        chain_artifact_type="infranym_text_render",
        rainbow_color_mnemonic_character_value="I",
        artifact_name="word_TEMPORAL",
        secret_word="TEMPORAL",
        image_text_style=ImageTextStyle.STATIC,
        # size defaults to (400, 200) - recommended!
    )
    text_path = text_render.encode()

    # Check required carrier size
    solution = "The discontinuity occurs at bar 77 where memory fragments collide"
    required_size = InfranymEncodedImageArtifact.calculate_required_carrier_size(
        text_path, len(solution)
    )
    print("💜 Text render: 400x200 (default)")
    print(f"💜 Required carrier: {required_size[0]}x{required_size[1]}")
    print("💜 NOTE: Your carrier MUST be at least this size!")

    # Step 2: Encode complete puzzle
    encoded = InfranymEncodedImageArtifact(
        base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
        thread_id="test_thread_002",
        chain_artifact_file_type="png",
        chain_artifact_type="infranym_encoded_image",
        rainbow_color_mnemonic_character_value="I",
        artifact_name="puzzle_card_23",
        carrier_image_path="/Volumes/LucidNonsense/White/tests/mocks/mock.png",
        text_render_path=text_path,
        surface_clue="Card 23 - Dewey Decimal 811.54 - Row 7",
        solution=solution,
        # NO size parameter - not a field on this artifact!
    )

    try:
        puzzle_path = encoded.encode()
        print(f"✅ Complete puzzle created: {puzzle_path}")

        # Test extraction
        print("\n🔍 Testing Layer 2 extraction...")
        text_img = encoded.extract_layer2_text()

        print("\n🔐 Testing Layer 3 decryption...")
        solution = encoded.solve_layer3()
        print(f"💜 Decrypted solution: {solution}")

        print(f"\n📄 Flattened: {encoded.flatten()}")

    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nSOLUTION:")
        print(f"  1. Make mock.png at least {required_size[0]}x{required_size[1]}")
        print("  2. OR use smaller text render: size=(200, 100)")
