from typing import List

from pydantic import BaseModel



class PulsarPalaceEffect(BaseModel):

    id: int

    def __init__(self, **data):
        super().__init__(**data)