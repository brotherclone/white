from pydantic import BaseModel


class PalaceRoom(BaseModel):

    id: int
    name: str
    description: str

    def __init__(self, **data):
        super().__init__(**data)
