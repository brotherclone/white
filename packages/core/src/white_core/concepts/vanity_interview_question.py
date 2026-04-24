from pydantic import BaseModel, Field


class VanityInterviewQuestion(BaseModel):
    number: int = Field(description="Question number (1-3)")
    question: str = Field(description="The actual question text")


class VanityInterviewQuestionOutput(BaseModel):
    questions: list[VanityInterviewQuestion] = Field(
        description="Three interview questions"
    )
