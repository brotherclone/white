import logging
import warnings

from enum import Enum
from types import MappingProxyType
from typing import Any
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


def _to_primitive(obj: Any):
    if obj is None:
        return None
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, MappingProxyType):
        return dict(obj)
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            data = obj.model_dump(exclude_none=True, mode="json")
        except TypeError:
            data = obj.model_dump(exclude_none=True, by_alias=True)
    elif isinstance(obj, (dict, list, tuple, str, int, float, bool)):
        data = obj
    elif hasattr(obj, "__dict__"):
        data = vars(obj)
    else:
        return str(obj)
    if isinstance(data, dict):
        return {k: _to_primitive(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_to_primitive(v) for v in data]
    return data


def save_markdown(
    content: str,
    path: str,
    append: bool = False,
    ensure_trailing_newline: bool = True,
    encoding: str = "utf-8",
) -> str:
    """
    Save `content` to a Markdown file at `path`.
    - `append`: if True, append to an existing file; otherwise overwrite.
    - `ensure_trailing_newline`: adds a final newline if missing.
    - Returns the absolute path to the written file as a string.
    """

    content = "" if content is None else str(content)
    if ensure_trailing_newline and not content.endswith("\n"):
        content += "\n"
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with p.open(mode, encoding=encoding, newline="\n") as f:
        f.write(content)
    return str(p.resolve())
