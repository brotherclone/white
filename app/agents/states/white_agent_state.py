from typing import Any, Dict, List, Optional, Annotated

from pydantic import BaseModel, ConfigDict, Field

from app.structures.concepts.facet_evolution import FacetEvolution
from app.structures.concepts.transformation_trace import TransformationTrace
from app.structures.enums.white_facet import WhiteFacet
from app.structures.manifests.song_proposal import SongProposal
from app.structures.artifacts.artifact_relationship import ArtifactRelationship
from app.structures.agents.base_rainbow_agent_state import dedupe_artifacts


def dedupe_traces(
    old_traces: List[TransformationTrace],
    new_traces: List[TransformationTrace],
) -> List[TransformationTrace]:
    """
    Deduplicate transformation traces by iteration_id to prevent exponential growth.

    Only adds NEW traces (those not already in old_traces).
    """
    if not new_traces:
        return old_traces or []
    if not old_traces:
        return new_traces

    existing_ids = {trace.iteration_id for trace in old_traces}
    unique_new = [
        trace for trace in new_traces if trace.iteration_id not in existing_ids
    ]

    return old_traces + unique_new


def dedupe_relationships(
    old_rels: List[ArtifactRelationship],
    new_rels: List[ArtifactRelationship],
) -> List[ArtifactRelationship]:
    """Deduplicate artifact relationships by artifact_id."""
    if not new_rels:
        return old_rels or []
    if not old_rels:
        return new_rels

    existing_ids = {rel.artifact_id for rel in old_rels}
    unique_new = [rel for rel in new_rels if rel.artifact_id not in existing_ids]

    return old_rels + unique_new


class MainAgentState(BaseModel):
    """
    Main state for The Prism (White Agent) - coordinator of chromatic rebracketing.

    The Prism doesn't generate chaos or methodology - it REVEALS what was always
    present by shifting the angle of perception through successive refraction.
    """

    # thread_id should persist once set - use reducer that keeps old value if new is empty/None
    thread_id: Annotated[str, lambda x, y: y if y else x]
    song_proposals: Annotated[SongProposal, lambda x, y: y or x] = Field(
        default_factory=lambda: SongProposal(iterations=[])
    )

    # Artifacts: Both ChainArtifact instances and dict representations
    # Rainbow agents serialize artifacts to dicts to avoid msgpack serialization issues
    # Use dedupe_artifacts to prevent exponential growth from node transitions
    artifacts: Annotated[List[Any], dedupe_artifacts] = Field(default_factory=list)

    # Workflow control
    workflow_paused: Annotated[bool, lambda x, y: y if y is not None else x] = False
    pause_reason: Annotated[Optional[str], lambda x, y: y or x] = None
    pending_human_action: Annotated[Optional[Dict[str, Any]], lambda x, y: y or x] = (
        None
    )

    # White Agent working variables (per-agent rebracketing)
    rebracketing_analysis: Annotated[Optional[str], lambda x, y: y or x] = None
    document_synthesis: Annotated[Optional[str], lambda x, y: y or x] = None

    meta_rebracketing: Annotated[Optional[str], lambda x, y: y or x] = (
        None  # The interference pattern across all seven lenses
    )
    chromatic_synthesis: Annotated[Optional[str], lambda x, y: y or x] = (
        None  # Final integration document
    )

    # White Facet system (cognitive lens)
    white_facet: Annotated[WhiteFacet | None, lambda x, y: y or x] = None
    white_facet_metadata: Annotated[str | Any, lambda x, y: y or x] = None
    facet_evolution: Annotated[Optional[FacetEvolution], lambda x, y: y or x] = None

    # Transformation traces (what boundaries shifted per agent)
    transformation_traces: Annotated[List[TransformationTrace], dedupe_traces] = Field(
        default_factory=list
    )

    # Artifact relationship graph
    artifact_relationships: Annotated[
        List[ArtifactRelationship], dedupe_relationships
    ] = Field(default_factory=list)

    # Agent readiness flags
    ready_for_red: Annotated[bool, lambda x, y: y if y is not None else x] = False
    ready_for_orange: Annotated[bool, lambda x, y: y if y is not None else x] = False
    ready_for_yellow: Annotated[bool, lambda x, y: y if y is not None else x] = False
    ready_for_green: Annotated[bool, lambda x, y: y if y is not None else x] = False
    ready_for_blue: Annotated[bool, lambda x, y: y if y is not None else x] = False
    ready_for_indigo: Annotated[bool, lambda x, y: y if y is not None else x] = False
    ready_for_violet: Annotated[bool, lambda x, y: y if y is not None else x] = False
    ready_for_white: Annotated[bool, lambda x, y: y if y is not None else x] = False

    # Workflow completion
    run_finished: Annotated[bool, lambda x, y: y if y is not None else x] = False

    # Execution mode controls (take new value if provided, else keep old)
    enabled_agents: Annotated[List[str], lambda x, y: y if y else x] = Field(
        default_factory=lambda: [
            "black",
            "red",
            "orange",
            "yellow",
            "green",
            "blue",
            "indigo",
            "violet",
        ]
    )
    stop_after_agent: Annotated[Optional[str], lambda x, y: y or x] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
