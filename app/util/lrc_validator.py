import os
import re
import sys

from typing import ClassVar, List, Tuple
from pydantic import BaseModel


class LRCValidator(BaseModel):
    """Validator for .lrc (Lyric) files"""

    title_re: ClassVar[re.Pattern] = re.compile(r"\[ti:\s*.+?\]", re.IGNORECASE)
    artist_re: ClassVar[re.Pattern] = re.compile(r"\[ar:\s*.+?\]", re.IGNORECASE)
    album_re: ClassVar[re.Pattern] = re.compile(r"\[al:\s*.+?\]", re.IGNORECASE)
    timestamp_pattern: ClassVar[re.Pattern] = re.compile(
        r"\[(\d{1,2}):(\d{2}(?:\.\d{1,3})?)\]"
    )
    # Known metadata tag prefixes (not timestamps)
    metadata_tags: ClassVar[set] = {"ti", "ar", "al", "by", "offset", "re", "ve"}
    # Pattern for bracket content that looks numeric (potential malformed timestamp)
    bracket_content_re: ClassVar[re.Pattern] = re.compile(r"\[([^\]]+)\]")

    # Max gap between consecutive timestamps before warning (seconds)
    max_gap_seconds: ClassVar[float] = 30.0

    def validate(self, lrc_content: str) -> Tuple[bool, List[str]]:
        """
        Validates an LRC file by checking required metadata, timestamp format,
        and timestamp sequencing.

        Args:
            lrc_content: Content of the LRC file as a string

        Returns:
            Tuple of (is_valid, error_messages)
        """
        err: List[str] = []
        warnings: List[str] = []

        lrc_content = lrc_content.lstrip("\ufeff")

        # Check for required metadata
        if not self.title_re.search(lrc_content):
            err.append("Missing title metadata [ti: title]")
        if not self.artist_re.search(lrc_content):
            err.append("Missing artist metadata [ar: artist]")
        if not self.album_re.search(lrc_content):
            err.append("Missing album metadata [al: album]")

        timestamps = []  # list of tuples (total_seconds, ts_str, line_num)
        for i, raw_line in enumerate(lrc_content.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue

            # Find all bracket-enclosed content on this line
            all_brackets = self.bracket_content_re.findall(line)
            valid_on_line = set()

            for m in self.timestamp_pattern.finditer(line):
                minutes = int(m.group(1))
                seconds = float(m.group(2))
                total_seconds = minutes * 60 + seconds
                timestamps.append((total_seconds, m.group(0), i))
                valid_on_line.add(m.group(0))

            # Check for malformed timestamps: bracket content that contains
            # digits and colons but didn't match the valid timestamp pattern
            for bracket in all_brackets:
                full = f"[{bracket}]"
                if full in valid_on_line:
                    continue

                # Skip known metadata tags
                tag = bracket.split(":")[0].strip().lower()
                if tag in self.metadata_tags:
                    continue

                # Check if it looks like it was meant to be a timestamp
                # (contains digits and any separator: colon, dot, slash, etc.)
                if re.search(r"\d", bracket) and re.search(r"[:./\\]\d", bracket):
                    # SMPTE-like: [MM:SS:FF.f] or [HH:MM:SS.f]
                    if re.match(r"\d{1,2}:\d{2}[:/]\d{1,2}", bracket):
                        err.append(
                            f"Line {i}: SMPTE/malformed timestamp {full}"
                            f" (has 3+ groups, expected [MM:SS.mmm])"
                        )
                    # Colon instead of dot before milliseconds: [MM:SS:mmm]
                    elif re.match(r"\d{1,2}:\d{2}:\d{1,3}$", bracket):
                        err.append(
                            f"Line {i}: Malformed timestamp {full}"
                            f" (colon before milliseconds, should be a dot)"
                        )
                    # Slash instead of dot: [MM:SS/mmm]
                    elif re.match(r"\d{1,2}:\d{2}/\d{1,3}$", bracket):
                        err.append(
                            f"Line {i}: Malformed timestamp {full}"
                            f" (slash before milliseconds, should be a dot)"
                        )
                    # Any other numeric bracket content that isn't a valid timestamp
                    elif re.match(r"[\d:./\\]+$", bracket):
                        err.append(f"Line {i}: Unrecognized timestamp format {full}")

        if not timestamps:
            err.append("No valid timestamp entries found")
        else:
            prev_time, prev_str, prev_line = -1.0, "[00:00.000]", 0
            for curr_time, curr_str, line_num in sorted(
                timestamps, key=lambda x: (x[2], x[0])
            ):
                if curr_time < prev_time:
                    err.append(
                        f"Non-sequential timestamp at line {line_num}: {curr_str}"
                        f" comes before {prev_str} at line {prev_line}"
                    )

                # Check for large time gaps
                gap = curr_time - prev_time
                if prev_time >= 0 and gap > self.max_gap_seconds:
                    warnings.append(
                        f"Line {line_num}: Large gap of {gap:.1f}s between"
                        f" {prev_str} (line {prev_line}) and {curr_str}"
                    )

                prev_time, prev_str, prev_line = curr_time, curr_str, line_num

        is_valid = len(err) == 0
        return is_valid, err + warnings


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python lrc_validator.py <directory_or_file_path>")
        sys.exit(1)

    path = sys.argv[1]
    validator = LRCValidator()

    targets: List[str] = []
    if os.path.isfile(path):
        if path.lower().endswith(".lrc"):
            targets = [path]
    else:
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(".lrc"):
                    targets.append(os.path.join(root, file))

    if not targets:
        print(f"No .lrc files found at ` {path} `")
        sys.exit(0)

    for file_path in targets:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"{file_path}: Failed to read file: {e}")
            continue

        is_valid, errors = validator.validate(content)
        if is_valid:
            print(f"{file_path}: Valid LRC file")
        else:
            print(f"{file_path}: Invalid LRC file")
            for error in errors:
                print(f"  - {error}")
