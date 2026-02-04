from typing import Optional, List, Dict, Any, Annotated
from pydantic import Field

from app.structures.agents.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.artifacts.alternate_timeline_artifact import (
    AlternateTimelineArtifact,
)
from app.structures.concepts.biographical_period import BiographicalPeriod
from app.structures.concepts.biographical_timeline import BiographicalTimeline
from app.structures.concepts.timeline_breakage_evaluation_results import (
    TimelineEvaluationResult,
)
from app.structures.artifacts.quantum_tape_label_artifact import (
    QuantumTapeLabelArtifact,
)
from app.structures.concepts.quantum_tape_musical_parameters import (
    QuantumTapeMusicalParameters,
)


def dedupe_periods(
    old_periods: List[BiographicalPeriod],
    new_periods: List[BiographicalPeriod],
) -> List[BiographicalPeriod]:
    """Deduplicate biographical periods by (start_date, end_date) pair."""
    if not new_periods:
        return old_periods or []
    if not old_periods:
        return new_periods

    existing_keys = {(p.start_date, p.end_date) for p in old_periods}
    unique_new = [
        p for p in new_periods if (p.start_date, p.end_date) not in existing_keys
    ]

    return old_periods + unique_new


class BlueAgentState(BaseRainbowAgentState):
    biographical_timeline: Annotated[
        Optional[BiographicalTimeline], lambda x, y: y or x
    ] = None
    forgotten_periods: Annotated[List[BiographicalPeriod], dedupe_periods] = Field(
        default_factory=list
    )
    selected_period: Annotated[
        Optional[BiographicalPeriod | Dict[str, Any]], lambda x, y: y or x
    ] = None
    selected_year: Annotated[Optional[int], lambda x, y: y or x] = (
        None  # Year key from biographical data
    )
    evaluation_result: Annotated[
        Optional[TimelineEvaluationResult], lambda x, y: y or x
    ] = None
    alternate_history: Annotated[
        Optional[AlternateTimelineArtifact], lambda x, y: y or x
    ] = None
    tape_label: Annotated[Optional[QuantumTapeLabelArtifact], lambda x, y: y or x] = (
        None
    )
    musical_params: Annotated[
        Optional[QuantumTapeMusicalParameters], lambda x, y: y or x
    ] = None
    iteration_count: Annotated[int, lambda x, y: y if y is not None else x] = 0
    max_iterations: Annotated[int, lambda x, y: y if y is not None else x] = 3
    biographical_data: Annotated[Optional[Dict[str, Any]], lambda x, y: y or x] = Field(
        default_factory=dict,
        description="Container for biographical data in agent state",
    )

    def __init__(self, **data):
        super().__init__(**data)
