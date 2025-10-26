import json
import warnings
import logging
from typing import Any

import yaml
from dotenv import load_dotenv

from app.agents.models.base_chain_artifact import ChainArtifact
from app.agents.models.text_chain_artifact_file import TextChainArtifactFile

load_dotenv()
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

def _to_primitive(obj: Any):
    if obj is None:
        return None
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
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
    yml_block = yaml.safe_dump(primitive, default_flow_style=False, sort_keys=False, allow_unicode=True)
    jsn_block = json.dumps(primitive, indent=2, ensure_ascii=False)
    ChainArtifact.save_md(
        artifact.base_path,
        artifact.artifact_name,
        f"```yml\n{yml_block}\n```",
        f"```json\n{jsn_block}\n```"
    )