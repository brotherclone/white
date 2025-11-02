from app.structures.artifacts.book_artifact import BookArtifact
from app.structures.artifacts.book_data import BookData
from app.structures.agents.base_rainbow_agent_state import BaseRainbowAgentState


class RedAgentState(BaseRainbowAgentState):

    main_generated_book: BookArtifact | None = None
    current_reaction_book: BookData | None = None
    should_respond_with_reaction_book: bool = False
    should_create_book: bool = True
    reaction_level: int = 0

    def __init__(self, **data):
        super().__init__(**data)