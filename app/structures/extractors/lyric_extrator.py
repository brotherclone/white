import os
from typing import Any, Dict, List

from dotenv import load_dotenv

from app.util.lrc_utils import load_lrc
from app.util.lrc_validator import LRCValidator
from app.util.manifest_loader import load_manifest


class LyricExtractor:
    """Standalone extractor for lyric data from LRC files"""

    lrc_path: str | None = None
    lyrics: List[Dict[str, Any]] = []

    def __init__(self, manifest_id: str):
        load_dotenv()
        self.manifest_id = manifest_id
        self.manifest_path = os.path.join(
            os.environ["MANIFEST_PATH"], manifest_id, f"{manifest_id}.yml"
        )

        if not os.path.exists(self.manifest_path):
            raise ValueError(f"Manifest file not found: {self.manifest_path}")

        # Load the manifest
        self.manifest = load_manifest(self.manifest_path)
        if self.manifest is None:
            raise ValueError("Manifest could not be loaded.")

        # Set up LRC path
        self.lrc_path = (
            os.path.join(
                os.environ["MANIFEST_PATH"], manifest_id, self.manifest.lrc_file
            )
            if hasattr(self.manifest, "lrc_file")
            else None
        )

        if self.lrc_path and os.path.isfile(self.lrc_path):
            with open(self.lrc_path, "r", encoding="utf-8") as f:
                lrc_content = f.read()
            is_valid, errors = LRCValidator().validate(lrc_content)
            if is_valid:
                print(f"{self.lrc_path}: Valid LRC file")
                self.lyrics = load_lrc(self.lrc_path)
            else:
                print(f"{self.lrc_path}: Invalid LRC file")
                for error in errors:
                    print(f" - {error}")
        else:
            print(f"LRC file not found or not specified: {self.lrc_path}")

    def extract_segment_features(
        self, lrc_path: str, start_time: float, end_time: float
    ) -> List[Dict[str, Any]]:
        """Extract lyric features for a specific time segment - matches the pattern of other extractors"""
        if not self.lyrics:
            # Load lyrics if not already loaded
            if os.path.isfile(lrc_path):
                self.lyrics = load_lrc(lrc_path)

        # Find lyrics that intersect with the time segment
        intersecting_lyrics = []
        for lyric in self.lyrics:
            lyric_start = lyric["start_time"]
            lyric_end = lyric["end_time"]

            # Check if lyric overlaps with segment
            if not (lyric_end <= start_time or lyric_start >= end_time):
                intersecting_lyrics.append(lyric)

        return intersecting_lyrics


if __name__ == "__main__":
    lyric_extractor = LyricExtractor(manifest_id="02_01")
