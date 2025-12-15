import re
from datetime import datetime, date
from textwrap import shorten
from typing import Any, Optional


def sanitize_for_filename(s: str, max_length: int = 50) -> str:
    """
    Sanitize a string to be safe for use in filenames.
    - Converts to lowercase
    - Replaces spaces with underscores
    - Removes or replaces invalid filename characters
    - Truncates to max_length
    """
    if not s:
        return "unnamed"
    # Replace spaces with underscores
    s = s.replace(" ", "_")
    # Remove any character that's not alphanumeric, underscore, hyphen, or period
    s = re.sub(r"[^a-zA-Z0-9_\-.]", "", s)
    # Convert to lowercase
    s = s.lower()
    # Truncate if too long
    if len(s) > max_length:
        s = s[:max_length]
    # Remove trailing underscores or hyphens
    s = s.rstrip("_-")
    return s if s else "unnamed"


def truncate_simple(s: str, n: int) -> str:
    """Simple slice: may cut a word in half."""
    return s if len(s) <= n else s[:n]


def truncate_with_ellipsis(s: str, n: int, ellipsis: str = "...") -> str:
    """Truncate and append ellipsis; respects the total length `n`."""
    if len(s) <= n:
        return s
    if n <= len(ellipsis):
        return ellipsis[:n]
    return s[: n - len(ellipsis)] + ellipsis


def truncate_word_safe(s: str, n: int, placeholder: str = "...") -> str:
    """Truncate on word boundaries using textwrap.shorten."""
    return shorten(s, width=n, placeholder=placeholder)


def resolve_name(value):
    """
    Return a safe name string for `value`.
    - If value is a list, use the first element.
    - If an element has `.name`, return that.
    - If an element is a dict with 'name', return that.
    - Otherwise return str(element) or empty string for None.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        value = value[0]
    if hasattr(value, "name"):
        try:
            return value.name or ""
        except ValueError as e:
            print(f"Error resolving name: {e}")
            return str(value)
    if isinstance(value, dict) and "name" in value:
        return value.get("name") or ""
    return str(value)


def format_date(value: Any, fmt: str = "%Y-%m-%d") -> Optional[str]:
    """
    Return a formatted date string, or None.
    - datetime -> ISO 8601 with time (use fmt to change)
    - date -> formatted with strftime(fmt)
    - ISO strings -> parsed via fromisoformat when possible
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()  # or value.strftime(fmt) for date-only
    if isinstance(value, date):
        return value.strftime(fmt)
    if isinstance(value, str):
        try:
            # accept ISO-formatted date/datetime strings
            dt = datetime.fromisoformat(value)
            return (
                dt.isoformat()
                if dt.time() != datetime.min.time()
                else dt.date().strftime(fmt)
            )
        except ValueError:
            return value  # leave as-is if parsing fails
    return None
