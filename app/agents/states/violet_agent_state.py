from typing import List, Optional, Annotated
from pydantic import Field

from app.structures.agents.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.artifacts.circle_jerk_interview_artifact import (
    CircleJerkInterviewArtifact,
)
from app.structures.concepts.Interview_item import InterviewItem
from app.structures.concepts.vanity_persona import VanityPersona


def safe_add(x, y):
    """Safely add two lists, handling None values"""
    if x is None and y is None:
        return None
    if x is None:
        return y
    if y is None:
        return x
    return x + y


class VioletAgentState(BaseRainbowAgentState):

    interview_collector: Annotated[Optional[List[InterviewItem]], safe_add] = Field(
        default_factory=list, description="Question and answer pairs for interview"
    )
    interviewer_persona: Annotated[VanityPersona, lambda x, y: y or x] = Field(
        ..., description="The persona of the interviewer"
    )
    interviewee_persona: Annotated[VanityPersona, lambda x, y: y or x] = Field(
        ..., description="The persona of the interviewee"
    )
    interviewer_objectives: Annotated[Optional[List[str]], safe_add] = Field(
        default_factory=list, description="The objectives of the interviewer"
    )
    interviewee_objectives: Annotated[Optional[List[str]], safe_add] = Field(
        default_factory=list, description="The objectives of the interviewee"
    )
    circle_jerk_interview: Annotated[
        Optional[CircleJerkInterviewArtifact], lambda x, y: y or x
    ] = Field(default=None, description="The circle jerk interview")

    def __init__(self, **data):
        super().__init__(**data)
