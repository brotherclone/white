from abc import ABC
from typing import Dict, List

from pydantic import Field

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.artifacts.last_human_artifact import LastHumanArtifact
from app.structures.artifacts.species_extinction_artifact import (
    SpeciesExtinctionArtifact,
)
from app.structures.concepts.last_human_species_extinction_parallel_moment import (
    LastHumanSpeciesExtinctionParallelMoment,
)


class LastHumanSpeciesExtinctionNarrativeArtifact(ChainArtifact, ABC):
    """
    Interleaved narrative showing species and human collapse in parallel.
    Structure: documentary-intimate duet.
    """

    # Source data
    species: SpeciesExtinctionArtifact
    human: LastHumanArtifact

    # Narrative arcs
    species_arc: str = Field(..., description="Dispassionate ecological documentation")
    human_arc: str = Field(..., description="Intimate personal experience")

    # Structure
    parallel_moments: List[LastHumanSpeciesExtinctionParallelMoment] = Field(
        default_factory=list, description="Key points where the narratives intersect"
    )

    # Tone
    elegiac_quality: str = Field(
        ..., description="How does this achieve memorial without nihilism?"
    )
    opening_image: str = Field(..., description="First visual/sensory detail")
    closing_image: str = Field(..., description="Final visual/sensory detail")

    # Musical mapping hints
    emotional_curve: List[str] = Field(
        default_factory=list,
        description="Trajectory: ['resignation', 'struggle', 'acceptance']",
    )
    silence_moments: List[str] = Field(
        default_factory=list, description="Where gaps/rests should emphasize loss"
    )

    def to_artifact_dict(self) -> Dict:
        """Serialize for ChainArtifact"""
        return {
            "species_name": self.species.common_name,
            "human_name": self.human.name,
            "species_arc": self.species_arc,
            "human_arc": self.human_arc,
            "parallel_moments": [m.model_dump() for m in self.parallel_moments],
            "elegiac_quality": self.elegiac_quality,
            "opening_image": self.opening_image,
            "closing_image": self.closing_image,
            "emotional_curve": self.emotional_curve,
            "silence_moments": self.silence_moments,
        }

    def to_markdown(self) -> str:
        """Format as readable markdown for artifact storage"""
        md = f"# {self.species.common_name} / {self.human.name}\n\n"
        md += f"## Species Arc: {self.species.common_name}\n{self.species_arc}\n\n"
        md += f"## Human Arc: {self.human.name}\n{self.human_arc}\n\n"

        if self.parallel_moments:
            md += "## Parallel Moments\n"
            for i, moment in enumerate(self.parallel_moments, 1):
                md += f"\n### {i}. {moment.timestamp_relative}\n"
                md += f"**Species:** {moment.species_moment}\n\n"
                md += f"**Human:** {moment.human_moment}\n\n"
                md += f"*Connection: {moment.thematic_connection}*\n"

        md += f"\n## Elegiac Quality\n{self.elegiac_quality}\n"

        return md
