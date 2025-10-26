"""Test utilities for normalizing mock YAML data before Pydantic instantiation.

Helpers convert bare enum-like tokens in the YAML fixtures (e.g. VANITY, RECONSTRUCTED)
into actual Enum members, and provide small helpers for common mock shapes like book_data and
text pages.
"""
from typing import Any, Dict, Type, Optional
from enum import Enum


def normalize_enum_field(d: Dict[str, Any], key: str, enum_cls: Type[Enum]) -> None:
    """If d[key] is a string, attempt to coerce it into an Enum member of enum_cls.

    Strategy (in order):
    - If value is already an instance of enum_cls, leave it.
    - Try to lookup by member name (exact, then uppercased).
    - Try to construct enum by value (enum_cls(value)).

    The function mutates ``d`` in-place and returns None.
    """
    if not isinstance(d, dict):
        return
    if key not in d:
        return
    val = d.get(key)
    # already normalized
    if isinstance(val, enum_cls):
        return
    if not isinstance(val, str):
        return
    raw = val.strip()
    # attempt lookup by name
    try:
        d[key] = enum_cls[raw]
        return
    except Exception:
        pass
    # try uppercased name (covers YAML that used lower/upper inconsistently)
    try:
        d[key] = enum_cls[raw.upper()]
        return
    except Exception:
        pass
    # attempt construction by value
    try:
        d[key] = enum_cls(raw)
        return
    except Exception:
        pass
    # last attempt: try stripping quotes or common whitespace; leave as-is if all fail
    return


def normalize_enum_fields(d: Dict[str, Any], mapping: Dict[str, Type[Enum]]) -> Dict[str, Any]:
    """Normalize multiple enum-like fields in a dict using a mapping of key->Enum.

    Mutates and returns the dict for convenience.
    """
    if not isinstance(d, dict):
        return d
    for k, enum_cls in mapping.items():
        try:
            normalize_enum_field(d, k, enum_cls)
        except Exception:
            # be forgiving in tests; leave original value to allow pydantic to raise if necessary
            continue
    return d


def normalize_book_data_enums(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize enum-like fields inside a top-level dict that may contain ``book_data``.

    Looks for a nested dict under the key ``book_data`` and normalizes its
    ``publisher_type`` and ``condition`` fields when present.
    """
    if not isinstance(data, dict):
        return data
    bd = data.get("book_data")
    if not isinstance(bd, dict):
        return data
    # import enums lazily to avoid import-time side effects during test collection
    try:
        from app.agents.enums.publisher_type import PublisherType
        from app.agents.enums.book_condition import BookCondition
    except Exception:
        return data

    normalize_enum_fields(bd, {
        "publisher_type": PublisherType,
        "condition": BookCondition,
    })
    data["book_data"] = bd
    return data


def normalize_bookdata_dict_only(bd: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize enum-like fields inside a BookData-shaped dict (no outer wrapper).

    This is a thin wrapper around :func:`normalize_enum_fields`.
    """
    if not isinstance(bd, dict):
        return bd
    try:
        from app.agents.enums.publisher_type import PublisherType
        from app.agents.enums.book_condition import BookCondition
    except Exception:
        return bd
    return normalize_enum_fields(bd, {
        "publisher_type": PublisherType,
        "condition": BookCondition,
    })


def normalize_text_page_defaults(d: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure minimal required fields for a TextChainArtifactFile fixture.

    - Sets a safe default for ``base_path`` when missing.
    - Attempts to coerce a simple ``rainbow_color`` dict into a ``RainbowTableColor`` instance
      if present (keeps failure silent so tests can still proceed).

    Returns the (possibly mutated) dict.
    """
    if not isinstance(d, dict):
        return d
    d.setdefault("base_path", "")
    # attempt to coerce rainbow_color if supplied as a small dict
    rc = d.get("rainbow_color") or d.get("rainbowColor")
    if isinstance(rc, dict) and rc:
        try:
            from app.structures.concepts.rainbow_table_color import RainbowTableColor
            d["rainbow_color"] = RainbowTableColor(**rc)
        except Exception:
            # be permissive in tests; leave as-is on failure
            pass
    return d
