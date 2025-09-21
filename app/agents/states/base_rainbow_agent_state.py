from typing import List
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class BaseRainbowAgentState(BaseModel):

    session_id: str | None = None
    timestamp: str | None = None
    messages: List[BaseMessage] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)


