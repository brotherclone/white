from typing import List, Any, Optional, Annotated

from pydantic import Field

from app.structures.agents.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.artifacts.newspaper_artifact import NewspaperArtifact
from app.structures.artifacts.symbolic_object_artifact import SymbolicObjectArtifact
from app.util.agent_state_utils import safe_add


class OrangeAgentState(BaseRainbowAgentState):

    synthesized_story: Annotated[Optional[NewspaperArtifact], lambda x, y: y or x] = (
        Field(default=None)
    )
    search_results: Annotated[Optional[List[Any]], safe_add] = Field(
        default=None, description="Web search results."
    )
    corpus_stories: Annotated[Optional[List[Any]], safe_add] = Field(
        default=None, description="Matching stories from the orange mythology corpus"
    )
    selected_story_id: Annotated[Optional[str], lambda x, y: y or x] = Field(
        default=None, description="ID of the selected story from the corpus"
    )
    symbolic_object: Annotated[
        Optional[SymbolicObjectArtifact], lambda x, y: y or x
    ] = Field(default=None, description="Custom symbolic object description")
    gonzo_perspective: Annotated[Optional[str], lambda x, y: y or x] = Field(
        default=None,
        description="Hunter S. Thompson style perspective on the story, such as a journalist, witness, authority, scholar",
    )
    gonzo_intensity: Annotated[int, lambda x, y: y if y is not None else x] = Field(
        default=3, description="How intense the gonzo perspective is"
    )
    mythologized_story: Annotated[Optional[NewspaperArtifact], lambda x, y: y or x] = (
        Field(
            default=None,
            description="Mythological story based on the story where the misremembered object has been added",
        )
    )

    def __init__(self, **data):
        super().__init__(**data)
