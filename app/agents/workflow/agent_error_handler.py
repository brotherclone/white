import json
import traceback
import logging

from functools import wraps

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def agent_error_handler(agent_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                state = None
                if "state" in kwargs:
                    state = kwargs.get("state")
                else:
                    for a in reversed(args):
                        if hasattr(a, "thread_id") or hasattr(a, "artifacts"):
                            state = a
                            break

                thread_id = (
                    getattr(state, "thread_id", "unknown")
                    if state is not None
                    else "unknown"
                )
                artifact_count = (
                    len(getattr(state, "artifacts", [])) if state is not None else 0
                )
                artifact_types = (
                    [type(a).__name__ for a in getattr(state, "artifacts", [])]
                    if state is not None
                    else []
                )
                state_keys = (
                    list(state.__dict__.keys())
                    if state is not None and hasattr(state, "__dict__")
                    else []
                )

                error_context = {
                    "agent": agent_name,
                    "thread_id": thread_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                    "state_keys": state_keys,
                    "artifact_count": artifact_count,
                    "artifact_types": artifact_types,
                }

                try:
                    import os

                    base_path = os.getenv(
                        "AGENT_WORK_PRODUCT_BASE_PATH", "chain_artifacts"
                    )
                    error_file = os.path.join(
                        base_path, thread_id, "debug", f"ERROR_{agent_name}.json"
                    )
                    os.makedirs(os.path.dirname(error_file), exist_ok=True)
                    with open(error_file, "w") as f:
                        json.dump(error_context, f, indent=2)
                except EnvironmentError as e:
                    logger.error(f"Could not write error context to file: {e!s}")
                    print(
                        "Could not write error context to file; continuing to print details"
                    )

                print(f"\n{'=' * 60}")
                print(f"\u274c ERROR in {agent_name}")
                print(f"{'=' * 60}")
                print(f"Type: {error_context['error_type']}")
                print(f"Message: {error_context['error_message']}")
                print(f"Context: thread_id={thread_id}, artifacts={artifact_count}")
                print(f"{'=' * 60}\n")

                raise  # Re-raise for LangGraph to handle

        return wrapper

    return decorator
