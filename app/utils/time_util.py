import datetime

def seconds_to_timedelta(seconds: float) -> datetime.timedelta:
    """
    Convert seconds to a timedelta object.

    Args:
        seconds (float): The time in seconds to convert.

    Returns:
        datetime.timedelta: The timedelta object representing the time.
    """
    return datetime.timedelta(seconds=seconds)

def timedelta_to_seconds(td: datetime.timedelta) -> float:
    """
    Convert a timedelta object to seconds.

    Args:
        td (datetime.timedelta): The timedelta object to convert.

    Returns:
        float: The total number of seconds represented by the timedelta.
    """
    return td.total_seconds()


def lrc_to_seconds(lrc_content: str) -> datetime.timedelta:
    """
    Convert LRC formatted lyrics to a timedelta object representing the total duration.

    Args:
        lrc_content (str): The content of the LRC file as a string.

    Returns:
        datetime.timedelta: The total duration represented by the LRC content.
    """
    total_seconds = 0
    for line in lrc_content.splitlines():
        if line.startswith('['):
            time_part = line.split(']')[0][1:]  # Extract time part
            minutes, seconds = map(float, time_part.split(':'))
            total_seconds += int(minutes * 60 + seconds)

    return datetime.timedelta(seconds=total_seconds)


def seconds_to_lrc(seconds: float) -> str:
    """
    Convert seconds to LRC formatted time string.

    Args:
        seconds (float): The time in seconds to convert.

    Returns:
        str: The LRC formatted time string.
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"[{minutes:02}:{secs:06.3f}]"

def get_duration(start_time:float | datetime.timedelta | str, end_time:float | datetime.timedelta | str) -> datetime.timedelta:
    """
    Calculate the duration between start and end times.

    Args:
        start_time (float | datetime.timedelta | str): The start time.
        end_time (float | datetime.timedelta | str): The end time.

    Returns:
        datetime.timedelta: The duration between start and end times.
    """
    if isinstance(start_time, str):
        start_time = float(start_time)
    if isinstance(end_time, str):
        end_time = float(end_time)

    if isinstance(start_time, datetime.timedelta):
        start_seconds = timedelta_to_seconds(start_time)
    else:
        start_seconds = start_time

    if isinstance(end_time, datetime.timedelta):
        end_seconds = timedelta_to_seconds(end_time)
    else:
        end_seconds = end_time

    return seconds_to_timedelta(end_seconds - start_seconds)

def convert_timedelta(obj):
    if isinstance(obj, datetime.timedelta):
        return obj.total_seconds()
    elif isinstance(obj, dict):
        return {k: convert_timedelta(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_timedelta(i) for i in obj]
    else:
        return obj