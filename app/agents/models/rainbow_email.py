from pydantic import BaseModel
from typing import List

class RainbowEmail(BaseModel):

    to: str
    frm: str
    subject: str
    body: str
    attachments: List[str] = []

    def __init__(self, **data):
        super().__init__(**data)
