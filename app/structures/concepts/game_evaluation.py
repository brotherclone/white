from pydantic import BaseModel, Field, field_validator


class GameEvaluationDecision(BaseModel):
    should_add_to_story: bool = Field(
        description="Should we add this game to the story by adding another room encounter?"
        " Or is the counter proposal and game run working nicely already?"
    )

    @field_validator("*", mode="before")
    def ensure_bool(cls, v):
        if isinstance(v, bool):
            return v
        raise ValueError("Boolean value required")
