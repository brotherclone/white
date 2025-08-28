import re
from typing import Tuple, List
from pydantic import BaseModel

class LRCValidator(BaseModel):
    """Validator for .lrc (Lyric) files"""

    def validate(self, content: str) -> Tuple[bool, List[str]]:
        """
        Validates an LRC file by checking required metadata and timestamp sequencing.

        Args:
            content: Content of the LRC file as a string

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check for required metadata
        if not re.search(r'\[ti:\s*.+?\]', content):
            errors.append("Missing title metadata [ti: title]")
        if not re.search(r'\[ar:\s*.+?\]', content):
            errors.append("Missing artist metadata [ar: artist]")
        if not re.search(r'\[al:\s*.+?\]', content):
            errors.append("Missing album metadata [al: album]")

        # Extract and validate timestamps
        timestamp_pattern = r'\[(\d{2}):(\d{2}\.\d{3})\]'
        timestamps = []

        for i, line in enumerate(content.splitlines()):
            match = re.match(timestamp_pattern, line)
            if match:
                minutes = int(match.group(1))
                seconds = float(match.group(2))
                total_seconds = minutes * 60 + seconds
                timestamps.append((total_seconds, match.group(0), i+1))

        if not timestamps:
            errors.append("No valid timestamp entries found")
        else:
            prev_time, prev_str, prev_line = -1, "[00:00.000]", 0

            for curr_time, curr_str, line_num in timestamps:
                if curr_time < prev_time:
                    errors.append(f"Non-sequential timestamp at line {line_num}: {curr_str} comes before {prev_str} at line {prev_line}")

                prev_time, prev_str, prev_line = curr_time, curr_str, line_num

        is_valid = len(errors) == 0
        return is_valid, errors

if __name__ == "__main__":
    v = LRCValidator()
    with open("../../staged_raw_material/03_03/03_03.lrc", "r", encoding="utf-8") as f:
        content = f.read()
        valid, errors = v.validate(content)

    # Optionally print results
    if not valid:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
    else:
        print("LRC file is valid")