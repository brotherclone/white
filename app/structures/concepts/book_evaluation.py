from pydantic import BaseModel, Field, field_validator, model_validator


class BookEvaluationDecision(BaseModel):
    """Decision about what to do next with books"""

    new_book: bool = Field(description="Should we generate a new book?")
    reaction_book: bool = Field(
        description="Should we generate a reaction book to an existing book?"
    )
    done: bool = Field(description="Are we done generating books?")

    @field_validator("*", mode="before")
    def ensure_bool(cls, v):
        if isinstance(v, bool):
            return v
        raise ValueError("Boolean value required")

    @model_validator(mode="after")
    def at_least_one_decision(cls, model):
        if not (model.new_book or model.reaction_book or model.done):
            raise ValueError("At least one decision must be made")
        return model
