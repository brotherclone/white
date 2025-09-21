import os
from typing import List, Dict, Any

from app.structures.extractors.base_manifest_extractor import BaseManifestExtractor
from app.util.lrc_utils import parse_lrc_time
from app.util.lrc_validator import LRCValidator
class LyricExtractor(BaseManifestExtractor):

    lrc_path: str | None = None
    lyrics: List[Dict[str, Any]] = []

    def __init__(self, **data):
        super().__init__(**data)
        self.lrc_path = (
                os.path.join(os.environ['MANIFEST_PATH'],
                self.manifest_id,
                self.manifest.lrc_file)
        ) if self.manifest_id else None
        if not self.lrc_path or not os.path.isfile(self.lrc_path):
            raise ValueError("lrc_path must be provided and point to a valid file.")
        with open(self.lrc_path, "r", encoding="utf-8") as f:
            lrc_content = f.read()
        is_valid, errors = LRCValidator().validate(lrc_content)
        if is_valid:
            print(f"{self.lrc_path}: Valid LRC file")
            self.lyrics = self.load_lrc(self.lrc_path)
        else:
            print(f"{self.lrc_path}: Invalid LRC file")
            for error in errors:
                print(f"  - {error}")

    @staticmethod
    def load_lrc(lrc_path: str) -> List[Dict[str, Any]]:
        """Parse LRC file into structured lyrical content (from original code)"""
        lyrics = []

        try:
            with open(lrc_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            print(f"Loading LRC file: {lrc_path}")
            print(f"Found {len(lines)} lines")

            current_timestamp = None
            current_time = None

            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                if line.startswith('[') and ']' in line and line.endswith(']'):
                    timestamp = line

                    if any(timestamp.startswith(f'[{tag}:') for tag in ['ti', 'ar', 'al', 'by', 'offset']):
                        print(f"Skipping metadata: {timestamp}")
                        continue

                    try:
                        parsed_time = parse_lrc_time(timestamp)
                        if parsed_time is None:
                            print(f"Warning: Could not parse timestamp {timestamp} on line {line_num + 1}")
                            continue

                        current_timestamp = timestamp
                        current_time = parsed_time
                        print(f"Found timestamp: {timestamp} = {parsed_time:.3f}s")

                    except Exception as e:
                        print(f"Error parsing timestamp {timestamp}: {e}")
                        continue

                elif line.startswith('[') and ']' in line:
                    bracket_end = line.find(']')
                    timestamp = line[:bracket_end + 1]
                    text = line[bracket_end + 1:].strip()

                    if any(timestamp.startswith(f'[{tag}:') for tag in ['ti', 'ar', 'al', 'by', 'offset']):
                        print(f"Skipping metadata: {timestamp}")
                        continue

                    try:
                        parsed_time = parse_lrc_time(timestamp)
                        if parsed_time is None:
                            print(f"Warning: Could not parse timestamp {timestamp} on line {line_num + 1}")
                            continue
                    except Exception as e:
                        print(f"Error parsing timestamp {timestamp}: {e}")
                        continue

                    if text:
                        lyrics.append({
                            'text': text,
                            'start_time': parsed_time,
                            'timestamp_raw': timestamp,
                            'line_number': line_num + 1
                        })
                        print(f"Added lyric: '{text}' at {parsed_time:.3f}s")

                else:
                    if current_time is not None and line:
                        lyrics.append({
                            'text': line,
                            'start_time': current_time,
                            'timestamp_raw': current_timestamp,
                            'line_number': line_num + 1
                        })
                        print(f"Added lyric: '{line}' at {current_time:.3f}s")

                        current_timestamp = None
                        current_time = None

            print(f"Successfully parsed {len(lyrics)} lyrical entries")

            # Calculate end times
            for i in range(len(lyrics)):
                if i < len(lyrics) - 1:
                    lyrics[i]['end_time'] = lyrics[i + 1]['start_time']
                else:
                    lyrics[i]['end_time'] = lyrics[i]['start_time'] + 3.0

            return lyrics

        except FileNotFoundError:
            print(f"ERROR: LRC file not found: {lrc_path}")
            return []
        except Exception as e:
            print(f"ERROR loading LRC file {lrc_path}: {e}")
            return []

if __name__ == "__main__":
    lyric_extractor = LyricExtractor(manifest_id="02_01")
