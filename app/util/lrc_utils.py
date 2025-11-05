import argparse
import os
import re
from typing import Any, Dict, List


def parse_lrc_time(time_str: str) -> float | None:
    """Parse LRC timestamp like [00:28.085] to seconds"""
    try:
        match = re.match(r"\[(\d{2}):(\d{2})\.(\d{3})]", time_str)
        if match:
            minutes, seconds, milliseconds = map(int, match.groups())
            return minutes * 60 + seconds + milliseconds / 1000
        else:
            print(f"Warning: Timestamp format not recognized: {time_str}")
            return None
    except Exception as e:
        print(f"Error parsing timestamp {time_str}: {e}")


def load_lrc(lrc_path: str) -> List[Dict[str, Any]]:
    """Parse LRC file into structured lyrical content (from original code)"""
    lyrics = []

    try:
        with open(lrc_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        print(f"Loading LRC file: {lrc_path}")
        print(f"Found {len(lines)} lines")

        current_timestamp = None
        current_time = None

        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            if line.startswith("[") and "]" in line and line.endswith("]"):
                timestamp = line

                if any(
                    timestamp.startswith(f"[{tag}:")
                    for tag in ["ti", "ar", "al", "by", "offset"]
                ):
                    print(f"Skipping metadata: {timestamp}")
                    continue

                try:
                    parsed_time = parse_lrc_time(timestamp)
                    if parsed_time is None:
                        print(
                            f"Warning: Could not parse timestamp {timestamp} on line {line_num + 1}"
                        )
                        continue

                    current_timestamp = timestamp
                    current_time = parsed_time
                    print(f"Found timestamp: {timestamp} = {parsed_time:.3f}s")

                except Exception as e:
                    print(f"Error parsing timestamp {timestamp}: {e}")
                    continue

            elif line.startswith("[") and "]" in line:
                bracket_end = line.find("]")
                timestamp = line[: bracket_end + 1]
                text = line[bracket_end + 1 :].strip()

                if any(
                    timestamp.startswith(f"[{tag}:")
                    for tag in ["ti", "ar", "al", "by", "offset"]
                ):
                    print(f"Skipping metadata: {timestamp}")
                    continue

                try:
                    parsed_time = parse_lrc_time(timestamp)
                    if parsed_time is None:
                        print(
                            f"Warning: Could not parse timestamp {timestamp} on line {line_num + 1}"
                        )
                        continue
                except Exception as e:
                    print(f"Error parsing timestamp {timestamp}: {e}")
                    continue

                if text:
                    lyrics.append(
                        {
                            "text": text,
                            "start_time": parsed_time,
                            "timestamp_raw": timestamp,
                            "line_number": line_num + 1,
                        }
                    )
                    print(f"Added lyric: '{text}' at {parsed_time:.3f}s")

            else:
                if current_time is not None and line:
                    lyrics.append(
                        {
                            "text": line,
                            "start_time": current_time,
                            "timestamp_raw": current_timestamp,
                            "line_number": line_num + 1,
                        }
                    )
                    print(f"Added lyric: '{line}' at {current_time:.3f}s")

                    current_timestamp = None
                    current_time = None

        print(f"Successfully parsed {len(lyrics)} lyrical entries")

        # Calculate end times
        for i in range(len(lyrics)):
            if i < len(lyrics) - 1:
                lyrics[i]["end_time"] = lyrics[i + 1]["start_time"]
            else:
                lyrics[i]["end_time"] = lyrics[i]["start_time"] + 3.0

        return lyrics

    except FileNotFoundError:
        print(f"ERROR: LRC file not found: {lrc_path}")
        return []
    except Exception as e:
        print(f"ERROR loading LRC file {lrc_path}: {e}")
        return []


def smpte_to_lrc_timestamp(smpte: str, fps: int = 30) -> str:
    """
    Convert SMPTE timecode (HH:MM:SS:FF.ff) to LRC timestamp [mm:ss.ccc].
    Assumes FF is frames, ff is subframes (fractional frames), default fps=30.
    """
    m = re.match(r"(\d{2}):(\d{2}):(\d{2}):(\d{2})\.(\d+)", smpte)
    if not m:
        return smpte  # Return unchanged if not SMPTE
    hh, mm, ss, ff, sub = map(int, m.groups())
    # Calculate total seconds
    frame_fraction = ff / fps + int(sub) / (fps * 10 ** len(str(sub)))
    total_seconds = hh * 3600 + mm * 60 + ss + frame_fraction
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - minutes * 60 - seconds) * 1000)
    return f"[{minutes:02d}:{seconds:02d}.{milliseconds:03d}]"


def convert_file_smpte_to_lrc_lines(lines, fps=30):
    """
    Convert all SMPTE timecodes in a list of lines to LRC timestamps.
    Returns a new list of lines.
    """
    out = []
    for line in lines:
        line = line.rstrip()
        m = re.match(r"(\d{2}:\d{2}:\d{2}:\d{2}\.\d+)", line)
        if m:
            out.append(smpte_to_lrc_timestamp(m.group(1), fps=fps))
        else:
            out.append(line)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Convert SMPTE timecodes in a file to LRC timestamps."
    )
    parser.add_argument("path", help="Path to input LRC or YAML file")
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second (default: 30)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: <input>.converted.lrc)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        print(f"ERROR: File not found: {args.path}")
        return

    with open(args.path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    converted = convert_file_smpte_to_lrc_lines(lines, fps=args.fps)

    out_path = args.output or f"{args.path}.converted.lrc"
    with open(out_path, "w", encoding="utf-8") as f:
        for line in converted:
            f.write(line + "\n")
    print(f"Converted file written to: {out_path}")


if __name__ == "__main__":
    main()
