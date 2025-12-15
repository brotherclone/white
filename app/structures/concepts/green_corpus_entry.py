from datetime import datetime
from typing import List, Dict
from pydantic import BaseModel, Field

from app.structures.artifacts.species_extinction_artifact import (
    SpeciesExtinctionArtifact,
)


class GreenCorpusEntry(BaseModel):
    """
    Single species entry in the Green Agent corpus.
    Combines all data needed for selection and narrative generation.
    """

    species: SpeciesExtinctionArtifact
    suggested_human_parallels: List[Dict[str, str]] = Field(
        default_factory=list, description="Pre-researched parallel suggestions"
    )
    musical_hints: Dict[str, str] = Field(
        default_factory=dict, description="Suggested musical mappings"
    )

    # Corpus management
    created_date: datetime = Field(default_factory=datetime.now)
    source_notes: str = Field(default="", description="Research sources")

    def to_artifact_dict(self) -> Dict:
        return {
            "species": self.species.to_artifact_dict(),
            "suggested_parallels": self.suggested_human_parallels,
            "musical_hints": self.musical_hints,
            "source_notes": self.source_notes,
        }
