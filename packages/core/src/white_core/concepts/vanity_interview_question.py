import json

from pydantic import BaseModel, Field, field_validator


class VanityInterviewQuestion(BaseModel):
    number: int = Field(description="Question number (1-3)")
    question: str = Field(description="The actual question text")


class VanityInterviewQuestionOutput(BaseModel):
    questions: list[VanityInterviewQuestion] = Field(
        description="Three interview questions"
    )

    @field_validator("questions", mode="before")
    @classmethod
    def _coerce_stringified_questions(cls, v):
        """The model occasionally calls this tool with the entire payload
        JSON-encoded as a string under this field (e.g.
        '{"questions": [{"number": 1, ...}]}') instead of passing the list
        directly. Unwrap that shape rather than fail validation outright."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
            except (ValueError, TypeError):
                return v
            if isinstance(parsed, dict) and "questions" in parsed:
                return parsed["questions"]
            if isinstance(parsed, list):
                return parsed
        return v
