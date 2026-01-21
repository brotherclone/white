from typing import List, Optional, Annotated
from pydantic import Field

from app.structures.agents.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.artifacts.circle_jerk_interview_artifact import (
    CircleJerkInterviewArtifact,
)
from app.structures.concepts.vanity_interview_question import VanityInterviewQuestion
from app.structures.concepts.vanity_interview_response import VanityInterviewResponse
from app.structures.concepts.vanity_persona import VanityPersona


class VioletAgentState(BaseRainbowAgentState):
    """
    State for Violet Agent's dialectical pressure-testing workflow.

    Workflow stages:
    1. select_persona → interviewer_persona
    2. generate_questions → interview_questions (List[InterviewQuestion])
    3. roll_for_hitl → needs_human_interview
    4. [human|simulated]_interview → interview_responses (List[InterviewResponse])
    5. synthesize_interview → circle_jerk_interview (artifact)
    6. generate_alternate_song_spec → counter_proposal
    """

    interviewer_persona: Annotated[Optional[VanityPersona], lambda x, y: y or x] = (
        Field(
            default=None,
            description="The randomly selected interviewer persona (name, type, publication)",
        )
    )
    interview_questions: Optional[List[VanityInterviewQuestion]] = Field(
        default=None,
        description="Three targeted questions from selected persona (structured)",
    )
    needs_human_interview: bool = Field(
        default=False, description="True if 9% roll succeeded - pause for real Gabe"
    )
    interview_responses: Optional[List[VanityInterviewResponse]] = Field(
        default=None,
        description="Three responses to questions (human or simulated, structured)",
    )
    circle_jerk_interview: Annotated[
        Optional[CircleJerkInterviewArtifact], lambda x, y: y or x
    ] = Field(default=None, description="Complete interview artifact with transcript")

    def __init__(self, **data):
        super().__init__(**data)
