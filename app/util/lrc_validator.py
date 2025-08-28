import re
import sys
import os
from typing import Tuple, List
from pydantic import BaseModel


class LRCValidator(BaseModel):
    """Validator for .lrc (Lyric) files"""

    def validate(self, lrc_content: str) -> Tuple[bool, List[str]]:
        """
        Validates an LRC file by checking required metadata and timestamp sequencing.

        Args:
            lrc_content: Content of the LRC file as a string

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check for required metadata
        if not re.search(r'\[ti:\s*.+?\]', lrc_content):
            errors.append("Missing title metadata [ti: title]")
        if not re.search(r'\[ar:\s*.+?\]', lrc_content):
            errors.append("Missing artist metadata [ar: artist]")
        if not re.search(r'\[al:\s*.+?\]', lrc_content):
            errors.append("Missing album metadata [al: album]")

        # Extract and validate timestamps
        timestamp_pattern = r'\[(\d{2}):(\d{2}\.\d{3})\]'
        timestamps = []

        for i, line in enumerate(lrc_content.splitlines()):
            match = re.match(timestamp_pattern, line)
            if match:
                minutes = int(match.group(1))
                seconds = float(match.group(2))
                total_seconds = minutes * 60 + seconds
                timestamps.append((total_seconds, match.group(0), i + 1))

        if not timestamps:
            errors.append("No valid timestamp entries found")
        else:
            prev_time, prev_str, prev_line = -1, "[00:00.000]", 0

            for curr_time, curr_str, line_num in timestamps:
                if curr_time < prev_time:
                    errors.append(
                        f"Non-sequential timestamp at line {line_num}: {curr_str} comes before {prev_str} at line {prev_line}")

                prev_time, prev_str, prev_line = curr_time, curr_str, line_num

        is_valid = len(errors) == 0
        return is_valid, errors


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python lrc_validator.py <directory_path>")
        sys.exit(1)
    directory_path = sys.argv[1]
    validator = LRCValidator()
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.lrc'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                is_valid, errors = validator.validate(content)
                if is_valid:
                    print(f"{file_path}: Valid LRC file")
                else:
                    print(f"{file_path}: Invalid LRC file")
                    for error in errors:
                        print(f"  - {error}")
