import re

def parse_lrc_time(time_str: str) -> float | None:
    """Parse LRC timestamp like [00:28.085] to seconds"""
    try:
        match = re.match(r'\[(\d{2}):(\d{2})\.(\d{3})]', time_str)
        if match:
            minutes, seconds, milliseconds = map(int, match.groups())
            return minutes * 60 + seconds + milliseconds / 1000
        else:
            print(f"Warning: Timestamp format not recognized: {time_str}")
            return None
    except Exception as e:
        print(f"Error parsing timestamp {time_str}: {e}")
        return None