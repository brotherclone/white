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