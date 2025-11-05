import json
import logging
import os
import warnings
from enum import Enum
from types import MappingProxyType
from typing import Any

import yaml
from dotenv import load_dotenv

from app.structures.artifacts.base_chain_artifact import ChainArtifact
from app.structures.artifacts.text_chain_artifact_file import \
    TextChainArtifactFile
from app.structures.concepts.rainbow_table_color import \
    the_rainbow_table_colors
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType

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


def save_artifact_to_md(artifact: TextChainArtifactFile):
    try:
        primitive = _to_primitive(artifact)
    except Exception as e:
        logging.error(f"Failed to convert artifact to primitive: {e}")
        primitive = str(artifact)
    yml_block = yaml.safe_dump(
        primitive, default_flow_style=False, sort_keys=False, allow_unicode=True
    )
    jsn_block = json.dumps(primitive, indent=2, ensure_ascii=False)
    ChainArtifact.save_md(
        artifact.base_path,
        artifact.artifact_name,
        f"```yml\n{yml_block}\n```",
        f"```json\n{jsn_block}\n```",
    )


def save_artifact_file_to_md(artifact: TextChainArtifactFile):
    """Write the artifact's text content to its artifact path, creating parent dirs as needed."""
    path_str = artifact.get_artifact_path(with_file_name=True)
    print(path_str)
    from pathlib import Path

    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(artifact.text_content or "")


if __name__ == "__main__":
    a = TextChainArtifactFile(
        base_path=f"{os.getenv('AGENT_WORK_PRODUCT_BASE_PATH')}/test",
        thread_id="123456",
        artifact_name="test",
        artifact_id="123456",
        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
        text_content="This is an example content for the artifact.",
        rainbow_color=the_rainbow_table_colors["R"],
    )
    save_artifact_file_to_md(a)
