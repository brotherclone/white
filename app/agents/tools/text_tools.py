import logging
import warnings

from enum import Enum
from types import MappingProxyType
from typing import Any
from dotenv import load_dotenv


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
