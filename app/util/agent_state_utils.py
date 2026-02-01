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
    """
    Safely merge two lists, taking new value if provided.

    NOTE: Previously this concatenated lists (x + y), causing exponential
    duplication in LangGraph workflows. Changed to replacement semantics
    since these fields represent "current results" not accumulated history.
    """
    if y is not None:
        return y
    return x


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


def _safe_serialize_state(state: Any) -> dict:
    """
    Safely serialize state, handling circular references and complex objects.
    """
    # Try model_dump with mode="json" first (handles Pydantic v2 better)
    if hasattr(state, "model_dump"):
        try:
            return state.model_dump(mode="json")
        except (ValueError, RecursionError) as e:
            logger.debug(f"model_dump failed: {e}, trying fallback")

    # Try .dict() for Pydantic v1 compatibility
    if hasattr(state, "dict"):
        try:
            return state.dict()
        except (ValueError, RecursionError) as e:
            logger.debug(f"dict() failed: {e}, trying fallback")

    # If state is already a dict, return it
    if isinstance(state, dict):
        return state

    # Fallback: extract basic state info without nested objects
    result = {}
    if hasattr(state, "__dict__"):
        for key, value in state.__dict__.items():
            if key.startswith("_"):
                continue
            try:
                # Try to JSON-serialize the value
                json.dumps(value, default=str)
                result[key] = value
            except (TypeError, ValueError, RecursionError):
                # If serialization fails, store a placeholder
                result[key] = f"<{type(value).__name__}: serialization failed>"

    return result


def get_state_snapshot(
    state: Any, node_name: str, thread_id: str, agent_name: str
) -> bool:
    load_dotenv()
    debug_dir = Path(os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")) / thread_id / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().isoformat()
    snapshot = debug_dir / f"{node_name}_{timestamp}.json"
    try:
        state_dict = _safe_serialize_state(state)
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
    except (ValueError, RecursionError) as e:
        # Final fallback: save minimal info
        with open(snapshot, "w") as f:
            json.dump(
                {
                    "node": node_name,
                    "timestamp": timestamp,
                    "agent_name": agent_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "thread_id": getattr(state, "thread_id", thread_id),
                    "artifact_count": len(getattr(state, "artifacts", [])),
                },
                f,
                indent=2,
            )
        logger.warning(f"Error saving state snapshot: {e}")
        return False
