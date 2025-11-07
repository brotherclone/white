from pydantic import BaseModel, Field


class YesOrNo(BaseModel):

    answer: bool = Field(description="Answer to the question.", default=False)

    def __init__(self, **data):
        super().__init__(**data)
