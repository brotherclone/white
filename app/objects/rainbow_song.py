from typing import Any

from pydantic import BaseModel


class RainbowSong(BaseModel):
   def __init__(self, /, **data: Any):
       super().__init__(**data)
