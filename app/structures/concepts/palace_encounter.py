from pydantic import BaseModel


class PalaceEncounter(BaseModel):

    name: str
    description: str

    def __init__(self, **data):
        super().__init__(**data)
