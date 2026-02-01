from typing import Any, Dict, List, Optional, Annotated
from pydantic import Field
from app.structures.agents.base_rainbow_agent_state import BaseRainbowAgentState


def dedupe_human_tasks(
    old_tasks: List[Dict[str, Any]],
    new_tasks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Deduplicate human tasks by task_id if present, otherwise by content equality."""
    if not new_tasks:
        return old_tasks or []
    if not old_tasks:
        return new_tasks

    # Build set of existing task IDs (if tasks have IDs) or use frozenset of items
    existing_keys = set()
    for task in old_tasks:
        if "task_id" in task:
            existing_keys.add(("id", task["task_id"]))
        else:
            # Use a hashable representation for comparison
            existing_keys.add(("hash", hash(frozenset(str(v) for v in task.values()))))

    unique_new = []
    for task in new_tasks:
        if "task_id" in task:
            key = ("id", task["task_id"])
        else:
            key = ("hash", hash(frozenset(str(v) for v in task.values())))
        if key not in existing_keys:
            unique_new.append(task)
            existing_keys.add(key)

    return old_tasks + unique_new


class BlackAgentState(BaseRainbowAgentState):
    """
    State for Black Agent workflow.

    Fields:
    - white_proposal: The specific iteration Black is responding to
    - song_proposals: Full negotiation history for context
    - counter_proposal: Black's generated response
    - artifacts: Generated sigils, EVPs, etc.
    """

    human_instructions: Annotated[Optional[str], lambda x, y: y or x] = ""
    pending_human_tasks: Annotated[List[Dict[str, Any]], dedupe_human_tasks] = Field(
        default_factory=list
    )
    awaiting_human_action: Annotated[bool, lambda x, y: y if y is not None else x] = (
        False
    )
    should_update_proposal_with_evp: Annotated[
        bool, lambda x, y: y if y is not None else x
    ] = False
    should_update_proposal_with_sigil: Annotated[
        bool, lambda x, y: y if y is not None else x
    ] = False

    def __init__(self, **data):
        super().__init__(**data)
