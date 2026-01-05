from typing import Optional, Annotated

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

    current_species: Annotated[
        Optional[SpeciesExtinctionArtifact], lambda x, y: y or x
    ] = Field(
        default=None,
        description="A randomly selected species to monitor to extinction.",
    )
    current_human: Annotated[Optional[LastHumanArtifact], lambda x, y: y or x] = Field(
        default=None, description="A randomly selected human to monitor to extinction."
    )
    current_parallel_moment: Annotated[
        Optional[LastHumanSpeciesExtinctionParallelMoment], lambda x, y: y or x
    ] = Field(
        default=None,
        description="Generated parallel moment for the extinct species and last human.",
    )
    current_narrative: Annotated[
        Optional[LastHumanSpeciesExtinctionNarrativeArtifact], lambda x, y: y or x
    ] = Field(
        default=None,
        description="A randomly selected narrative to monitor to extinction.",
    )
    current_survey: Annotated[
        Optional[ArbitrarysSurveyArtifact], lambda x, y: y or x
    ] = Field(
        default=None,
        description="Surveys from the Culture ship Arbitrary or its sub-instance",
    )
    current_decision: Annotated[
        Optional[RescueDecisionArtifact], lambda x, y: y or x
    ] = Field(
        default=None,
        description="Claude's, or what its evolved to, decision made by the agent to rescue the last human.",
    )

    def __init__(self, **data):
        super().__init__(**data)
