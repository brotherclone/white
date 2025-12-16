from typing import Optional, List
from pydantic import Field

from app.structures.agents.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.artifacts.alternate_timeline_artifact import (
    AlternateTimelineArtifact,
)
from app.structures.concepts.biographical_period import BiographicalPeriod
from app.structures.concepts.biographical_timeline import BiographicalTimeline
from app.structures.artifacts.quantum_tape_label_artifact import (
    QuantumTapeLabelArtifact,
)
from app.structures.concepts.quantum_tape_musical_parameters import (
    QuantumTapeMusicalParameters,
)


class BlueAgentState(BaseRainbowAgentState):
    biographical_timeline: Optional[BiographicalTimeline] = None
    forgotten_periods: List[BiographicalPeriod] = Field(default_factory=list)
    selected_period: Optional[BiographicalPeriod] = None
    alternate_history: Optional[AlternateTimelineArtifact] = None
    tape_label: Optional[QuantumTapeLabelArtifact] = None
    musical_params: Optional[QuantumTapeMusicalParameters] = None
    iteration_count: int = 0

    def __init__(self, **data):
        super().__init__(**data)
