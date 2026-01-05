from typing import Annotated
from app.structures.agents.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.artifacts.book_artifact import BookArtifact


class RedAgentState(BaseRainbowAgentState):

    main_generated_book: Annotated[BookArtifact | None, lambda x, y: y or x] = None
    current_reaction_book: Annotated[BookArtifact | None, lambda x, y: y or x] = None
    should_respond_with_reaction_book: Annotated[
        bool, lambda x, y: y if y is not None else x
    ] = False
    should_create_book: Annotated[bool, lambda x, y: y if y is not None else x] = True
    reaction_level: Annotated[int, lambda x, y: y if y is not None else x] = 0

    def __init__(self, **data):
        super().__init__(**data)
