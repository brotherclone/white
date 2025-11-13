"""Test utilities for normalizing mock YAML data before Pydantic instantiation.

Helpers convert bare enum-like tokens in the YAML fixtures (e.g., VANITY, RECONSTRUCTED)
into actual Enum members and provide small helpers for common mock shapes like book_data and
text pages.
"""

from enum import Enum
from typing import Any, Dict, Type

from app.structures.concepts.rainbow_table_color import RainbowTableColor
from app.structures.enums.book_condition import BookCondition
from app.structures.enums.publisher_type import PublisherType


def normalize_enum_field(d: Dict[str, Any], key: str, enum_cls: Type[Enum]) -> None:
    """If d[key] is a string, attempt to coerce it into an Enum member of enum_cls.

    Strategy (in order):
    - If value is already an instance of enum_cls, leave it.
    - Try to look up by member name (exact, then uppercased).
    - Try to construct enum by value (enum_cls(value)).

    The function mutates ``d`` in-place and returns None.
    """
    if not isinstance(d, dict):
        return
    if key not in d:
        return
    val = d.get(key)
    if isinstance(val, enum_cls):
        return
    if not isinstance(val, str):
        return
    raw = val.strip()
    try:
        d[key] = enum_cls[raw]
        return
    except (KeyError, ValueError) as e:
        print(f"Failed to coerce {key} to {enum_cls}: {e}")
        pass
    try:
        d[key] = enum_cls[raw.upper()]
        return
    except (KeyError, ValueError) as e:
        print(f"Failed to coerce {key} to {enum_cls}: {e}")
        pass
    try:
        d[key] = enum_cls(raw)
        return
    except ValueError as e:
        print(f"Failed to coerce {key} to {enum_cls}: {e}")
        pass
    return


def normalize_enum_fields(
    d: Dict[str, Any], mapping: Dict[str, Type[Enum]]
) -> Dict[str, Any]:
    """Normalize multiple enum-like fields in a dict using a mapping of key->Enum.

    Mutates and returns the dict for convenience.
    """
    if not isinstance(d, dict):
        return d
    for k, enum_cls in mapping.items():
        try:
            normalize_enum_field(d, k, enum_cls)
        except ValueError:
            print(f"Failed to coerce {k} to {enum_cls}")
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

    normalize_enum_fields(
        bd,
        {
            "publisher_type": PublisherType,
            "condition": BookCondition,
        },
    )
    data["book_data"] = bd
    return data


def normalize_book_data_dict_only(bd: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize enum-like fields inside a BookData-shaped dict (no outer wrapper).

    This is a thin wrapper around: func:`normalize_enum_fields`.
    """
    if not isinstance(bd, dict):
        return bd
    try:
        from app.structures.enums.book_condition import BookCondition
        from app.structures.enums.publisher_type import PublisherType
    except ValueError:
        print("Failed to import enums; skipping bookdata enum normalization")
        return bd
    return normalize_enum_fields(
        bd,
        {
            "publisher_type": PublisherType,
            "condition": BookCondition,
        },
    )


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
    rc = d.get("rainbow_color") or d.get("rainbowColor")
    if isinstance(rc, dict) and rc:
        try:
            d["rainbow_color"] = RainbowTableColor(**rc)
        except ValueError:
            print(f"Failed to coerce rainbow_color dict to RainbowTableColor: {rc}")
            pass
    return d


def normalize_newspaper_fixture(d: dict) -> dict:
    """Normalize newspaper fixture dicts for test instantiation.

    - Ensure `tags` is a list (defaults to empty list).
    - If `page` is a list containing a dict (common in fixtures), unwrap the first
      element, apply text-page defaults and convert to TextChainArtifactFile.
    """
    if not isinstance(d, dict):
        return d

    # normalize tags
    if "tags" not in d or d.get("tags") is None:
        d["tags"] = []
    else:
        if not isinstance(d["tags"], list):
            d["tags"] = list(d["tags"])

    # normalize page -> accept dict or single-item list-of-dict
    page = d.get("page")

    # unwrap common case where page is a list with one dict item
    if isinstance(page, list) and len(page) > 0:
        first = page[0]
        if isinstance(first, dict):
            page = first
        else:
            # leave non-dict lists as-is
            page = first

    if isinstance(page, dict):
        page = normalize_text_page_defaults(page)
        try:
            from app.structures.artifacts.text_artifact_file import (
                TextChainArtifactFile,
            )

            d["page"] = TextChainArtifactFile(**page)
        except Exception:
            # keep the dict if conversion fails; tests should still proceed
            d["page"] = page
    else:
        # leave non-dict/non-list pages as-is (None, string, etc.)
        d["page"] = page

    return d
