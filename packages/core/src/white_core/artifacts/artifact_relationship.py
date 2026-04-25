from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field


class ArtifactRelationship(BaseModel):
    """
    Semantic relationships between artifacts beyond simple type filtering.

    Artifacts can resonate across agents, entangle with each other,
    or reveal temporal depth across the chromatic spectrum.
    """

    artifact_id: str
    resonant_agents: List[str] = Field(
        default_factory=list
    )  # Which agents this artifact speaks to
    entangled_with: List[str] = Field(default_factory=list)  # Other artifact IDs
    temporal_depth: Dict[str, str] = Field(
        default_factory=dict
    )  # Meanings at different spectrum points
    semantic_tags: List[str] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


if __name__ == "__main__":
    mock_relationship = ArtifactRelationship(
        artifact_id="mock_artifact",
        resonant_agents=["agent1", "agent2"],
        entangled_with=["artifact1", "artifact2"],
        temporal_depth={"red": "meaning1", "blue": "meaning2"},
        semantic_tags=["tag1", "tag2"],
    )
    print(mock_relationship)
