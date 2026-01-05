from typing import Any, Dict, List, Optional

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
    song_proposals: SongProposal = Field(
        default_factory=lambda: SongProposal(iterations=[])
    )

    # Artifacts: Both ChainArtifact instances and dict representations
    # Rainbow agents serialize artifacts to dicts to avoid msgpack serialization issues
    artifacts: List[Any] = Field(default_factory=list)

    # Workflow control
    workflow_paused: bool = False
    pause_reason: Optional[str] = None
    pending_human_action: Optional[Dict[str, Any]] = None

    # White Agent working variables (per-agent rebracketing)
    rebracketing_analysis: Optional[str] = None
    document_synthesis: Optional[str] = None

    meta_rebracketing: Optional[str] = (
        None  # The interference pattern across all seven lenses
    )
    chromatic_synthesis: Optional[str] = None  # Final integration document

    # White Facet system (cognitive lens)
    white_facet: WhiteFacet | None = None
    white_facet_metadata: str | Any = None
    facet_evolution: Optional[FacetEvolution] = None

    # Transformation traces (what boundaries shifted per agent)
    transformation_traces: List[TransformationTrace] = Field(default_factory=list)

    # Artifact relationship graph
    artifact_relationships: List[ArtifactRelationship] = Field(default_factory=list)

    # Agent readiness flags
    ready_for_red: bool = False
    ready_for_orange: bool = False
    ready_for_yellow: bool = False
    ready_for_green: bool = False
    ready_for_blue: bool = False
    ready_for_indigo: bool = False
    ready_for_violet: bool = False
    ready_for_white: bool = False

    # Workflow completion
    run_finished: bool = False

    # Execution mode controls
    enabled_agents: List[str] = Field(
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
    stop_after_agent: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
