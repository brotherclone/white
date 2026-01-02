from typing import List, Any, Optional

from pydantic import Field

from app.structures.agents.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.artifacts.newspaper_artifact import NewspaperArtifact
from app.structures.artifacts.symbolic_object_artifact import SymbolicObjectArtifact


class OrangeAgentState(BaseRainbowAgentState):

    synthesized_story: Optional[NewspaperArtifact] = Field(default=None)
    search_results: Optional[List[Any]] = Field(
        default=None, description="Web search results."
    )
    corpus_stories: Optional[List[Any]] = Field(
        default=None, description="Matching stories from the orange mythology corpus"
    )
    selected_story_id: Optional[str] = Field(
        default=None, description="ID of the selected story from the corpus"
    )
    symbolic_object: Optional[SymbolicObjectArtifact] = Field(
        default=None, description="Custom symbolic object description"
    )
    gonzo_perspective: Optional[str] = Field(
        default=None,
        description="Hunter S. Thompson style perspective on the story, such as a journalist, witness, authority, scholar",
    )
    gonzo_intensity: int = Field(
        default=3, description="How intense the gonzo perspective is"
    )
    mythologized_story: Optional[NewspaperArtifact] = Field(
        default=None,
        description="Mythological story based on the story where the misremembered object has been added",
    )

    def __init__(self, **data):
        super().__init__(**data)
