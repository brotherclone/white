from typing import List, Optional
from pydantic import Field

from app.structures.agents.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.artifacts.circle_jerk_interview_artifact import (
    CircleJerkInterviewArtifact,
)
from app.structures.concepts.Interview_item import InterviewItem
from app.structures.concepts.vanity_persona import VanityPersona


class VioletAgentState(BaseRainbowAgentState):

    interview_collector: Optional[List[InterviewItem]] = Field(
        default_factory=list, description="Question and answer pairs for interview"
    )
    interviewer_persona: VanityPersona = Field(
        ..., description="The persona of the interviewer"
    )
    interviewee_persona: VanityPersona = Field(
        ..., description="The persona of the interviewee"
    )
    interviewer_objectives: Optional[List[str]] = Field(
        default_factory=list, description="The objectives of the interviewer"
    )
    interviewee_objectives: Optional[List[str]] = Field(
        default_factory=list, description="The objectives of the interviewee"
    )
    circle_jerk_interview: Optional[CircleJerkInterviewArtifact] = Field(
        default=None, description="The circle jerk interview"
    )

    def __init__(self, **data):
        super().__init__(**data)
