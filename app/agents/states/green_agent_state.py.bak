from typing import Optional

from pydantic import Field

from app.structures.agents.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.artifacts.arbitrarys_survey_artifact import ArbitrarysSurveyArtifact
from app.structures.artifacts.last_human_artifact import LastHumanArtifact
from app.structures.artifacts.last_human_species_extinction_narative_artifact import (
    LastHumanSpeciesExtinctionNarrativeArtifact,
)
from app.structures.artifacts.rescue_decision_artifact import RescueDecisionArtifact
from app.structures.artifacts.species_extinction_artifact import (
    SpeciesExtinctionArtifact,
)
from app.structures.concepts.last_human_species_extinction_parallel_moment import (
    LastHumanSpeciesExtinctionParallelMoment,
)


class GreenAgentState(BaseRainbowAgentState):

    current_species: Optional[SpeciesExtinctionArtifact] = Field(
        default=None,
        description="A randomly selected species to monitor to extinction.",
    )
    current_human: Optional[LastHumanArtifact] = Field(
        default=None, description="A randomly selected human to monitor to extinction."
    )
    current_parallel_moment: Optional[LastHumanSpeciesExtinctionParallelMoment] = Field(
        default=None,
        description="Generated parallel moment for the extinct species and last human.",
    )
    current_narrative: Optional[LastHumanSpeciesExtinctionNarrativeArtifact] = Field(
        default=None,
        description="A randomly selected narrative to monitor to extinction.",
    )
    current_survey: Optional[ArbitrarysSurveyArtifact] = Field(
        default=None,
        description="Surveys from the Culture ship Arbitrary or its sub-instance",
    )
    current_decision: Optional[RescueDecisionArtifact] = Field(
        default=None,
        description="Claude's, or what its evolved to, decision made by the agent to rescue the last human.",
    )

    def __init__(self, **data):
        super().__init__(**data)
