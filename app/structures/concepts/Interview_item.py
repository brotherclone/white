from typing import Optional

from pydantic import Field, BaseModel


class InterviewItem(BaseModel):

    question: str = Field(..., description="The interview question")
    answer: Optional[str] = Field(
        default=None, description="The answer to the interview question"
    )
