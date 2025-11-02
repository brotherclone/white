from textwrap import shorten

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