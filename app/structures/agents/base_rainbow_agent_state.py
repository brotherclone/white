from typing import Any, List, Optional, Annotated
from operator import add

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from app.structures.manifests.song_proposal import SongProposal, SongProposalIteration


def _get_artifact_id(art: Any) -> Optional[str]:
    """Extract artifact_id from an artifact (dict or object)."""
    if isinstance(art, dict):
        return art.get("artifact_id")
    return getattr(art, "artifact_id", None)


def dedupe_artifacts(old: List[Any], new: List[Any]) -> List[Any]:
    """
    Merge artifact lists, deduplicating by artifact_id.

    This prevents the exponential artifact duplication caused by nodes
    that mutate state.artifacts and return the full state. With the
    standard `add` reducer, each node transition would double the list.
    """
    if not new:
        return old if old else []
    if not old:
        return new

    # Build set of existing artifact IDs
    seen_ids = set()
    for art in old:
        art_id = _get_artifact_id(art)
        if art_id:
            seen_ids.add(art_id)

    # Start with old artifacts, add only new unique ones
    result = list(old)
    for art in new:
        art_id = _get_artifact_id(art)
        if art_id and art_id not in seen_ids:
            result.append(art)
            seen_ids.add(art_id)
        elif not art_id:
            # If no artifact_id, add it anyway (fallback behavior)
            result.append(art)

    return result


class BaseRainbowAgentState(BaseModel):
    """
    Base state for Rainbow Table agents.

    All fields are Annotated to allow multiple nodes to write concurrently in LangGraph.
    - Single-value fields use `lambda x, y: y or x` (take new value if present, else keep old)
    - List fields use `add` operator (concatenate lists) or custom reducers
    """

    # Scalar fields - use "take last non-None" reducer
    session_id: Annotated[Optional[str], lambda x, y: y or x] = None
    timestamp: Annotated[Optional[str], lambda x, y: y or x] = None
    thread_id: Annotated[Optional[str], lambda x, y: y or x] = Field(
        default=None, description="Unique ID of the thread."
    )
    white_proposal: Annotated[Optional[SongProposalIteration], lambda x, y: y or x] = (
        None
    )
    song_proposals: Annotated[Optional[SongProposal], lambda x, y: y or x] = None
    counter_proposal: Annotated[
        Optional[SongProposalIteration], lambda x, y: y or x
    ] = None
    messages: Annotated[List[BaseMessage], add] = Field(default_factory=list)
    # Use dedupe_artifacts to prevent exponential growth from nodes that
    # mutate state.artifacts and return the full state
    artifacts: Annotated[List[Any], dedupe_artifacts] = Field(default_factory=list)
    # Deduplicate skipped nodes to prevent exponential growth
    skipped_nodes: Annotated[
        List[str], lambda old, new: list(dict.fromkeys((old or []) + (new or [])))
    ] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
