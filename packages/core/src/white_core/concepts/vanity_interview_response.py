from pydantic import BaseModel, Field


class VanityInterviewResponse(BaseModel):
    question_number: int = Field(description="Which question this answers")
    response: str = Field(description="The actual response text")


class VanityInterviewResponseOutput(BaseModel):
    responses: list[VanityInterviewResponse] = Field(
        description="Three interview responses"
    )
