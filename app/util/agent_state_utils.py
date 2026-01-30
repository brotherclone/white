import datetime
import json
import os
import logging
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def safe_add(x, y):
    """Safely add two lists, handling None values"""
    if x is None and y is None:
        return None
    if x is None:
        return y
    if y is None:
        return x
    return x + y


def validate_state_structure(state) -> dict:
    issues = []
    for field_name, field_value in state.__dict__.items():
        try:
            json.dumps(field_value, default=str)
        except Exception as e:
            issues.append(
                {
                    "field": field_name,
                    "type": type(field_value).__name__,
                    "error": str(e),
                }
            )
    for artifact in state.artifacts:
        try:
            artifact.dict()  # Pydantic serialization
        except Exception as e:
            issues.append({"artifact": artifact.__class__.__name__, "error": str(e)})

    return {"valid": len(issues) == 0, "issues": issues}


def get_state_snapshot(
    state: Any, node_name: str, thread_id: str, agent_name: str
) -> bool:
    load_dotenv()
    debug_dir = Path(os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")) / thread_id / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().isoformat()
    snapshot = debug_dir / f"{node_name}_{timestamp}.json"
    try:
        state_dict = state.dict() if hasattr(state, "dict") else state
        with open(snapshot, "w") as f:
            json.dump(
                {
                    "node": node_name,
                    "timestamp": timestamp,
                    "state": state_dict,
                    "agent_name": agent_name,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"✅ State snapshot saved: {snapshot}")
        return True
    except ValueError as e:
        with open(snapshot, "w") as f:
            json.dump(
                {
                    "node": node_name,
                    "timestamp": timestamp,
                    "agent_name": agent_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                f,
                indent=2,
            )
        logger.warning(f"Error saving state snapshot: {e}")
        return False
