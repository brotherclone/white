from pydantic import BaseModel

class YesOrNo(BaseModel):

    answer: bool

    def __init__(self, **data):
        super().__init__(**data)