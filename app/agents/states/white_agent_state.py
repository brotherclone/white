from typing import Any, Dict, List, Optional, Annotated
from operator import add  # Added by auto_annotate

from pydantic import BaseModel, ConfigDict, Field

from app.structures.concepts.facet_evolution import FacetEvolution
from app.structures.concepts.transformation_trace import TransformationTrace
from app.structures.enums.white_facet import WhiteFacet
from app.structures.manifests.song_proposal import SongProposal
from app.structures.artifacts.artifact_relationship import ArtifactRelationship


class MainAgentState(BaseModel):
    """
    Main state for The Prism (White Agent) - coordinator of chromatic rebracketing.

    The Prism doesn't generate chaos or methodology - it REVEALS what was always
    present by shifting the angle of perception through successive refraction.
    """

    thread_id: str
    song_proposals: Annotated[SongProposal, lambda x, y: y or x] = Field(
        default_factory=lambda: SongProposal(iterations=[])
    )

    # Artifacts: Both ChainArtifact instances and dict representations
    # Rainbow agents serialize artifacts to dicts to avoid msgpack serialization issues
    artifacts: Annotated[List[Any], add] = Field(default_factory=list)

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
    transformation_traces: Annotated[List[TransformationTrace], add] = Field(
        default_factory=list
    )

    # Artifact relationship graph
    artifact_relationships: Annotated[List[ArtifactRelationship], add] = Field(
        default_factory=list
    )

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

    # Execution mode controls
    enabled_agents: Annotated[List[str], add] = Field(
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
