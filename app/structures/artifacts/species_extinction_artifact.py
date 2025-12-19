import os
import yaml

from abc import ABC
from pathlib import Path
from typing import List, Optional, Literal, Dict
from pydantic import Field
from dotenv import load_dotenv

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.concepts.population_data import PopulationData
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType
from app.structures.enums.extinction_cause import ExtinctionCause
from app.util.string_utils import sanitize_for_filename

load_dotenv()


class SpeciesExtinctionArtifact(ChainArtifact, ABC):
    """
    Core species extinction data model.
    Combines IUCN-style scientific data with narrative hooks.
    """

    # ToDo: Needs a for_prompt() method

    chain_artifact_type: ChainArtifactType = ChainArtifactType.SPECIES_EXTINCTION
    chain_artifact_file_type: ChainArtifactFileType = ChainArtifactFileType.YML
    rainbow_color_mnemonic_character_value: str = "G"

    # Scientific identification
    scientific_name: str = Field(..., description="Binomial nomenclature")
    common_name: str = Field(..., description="Common/vernacular name")
    taxonomic_group: str = Field(..., description="e.g., mammal, bird, insect, marine")

    # Conservation status
    iucn_status: str = Field(..., description="IUCN Red List category")
    extinction_year: int = Field(
        ..., ge=2020, le=2150, description="Projected/actual extinction year"
    )
    population_trajectory: List[PopulationData] = Field(default_factory=list)

    # Geographic
    habitat: str = Field(..., description="Primary habitat/location")
    endemic: bool = Field(default=False, description="Found only in one location")
    range_km2: Optional[float] = Field(None, description="Historical range in km²")

    # Extinction drivers
    primary_cause: ExtinctionCause
    secondary_causes: List[str] = Field(default_factory=list)
    anthropogenic_factors: List[str] = Field(default_factory=list)

    # Ecological impact
    cascade_effects: List[str] = Field(
        default_factory=list, description="Downstream ecological consequences"
    )
    ecosystem_role: str = Field(..., description="Ecological niche/function")

    # Narrative hooks
    symbolic_resonance: List[str] = Field(
        default_factory=list, description="Why this species matters narratively"
    )
    human_parallel_hints: List[str] = Field(
        default_factory=list, description="Suggested human condition parallels"
    )

    # Scoring
    narrative_potential_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How rich is this for storytelling?"
    )
    symbolic_weight: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Cultural/emotional resonance"
    )

    # Physical characteristics (for musical mapping)
    size_category: Literal["tiny", "small", "medium", "large", "massive"] = "medium"
    lifespan_years: Optional[int] = None
    movement_pattern: Optional[str] = Field(
        None, description="migratory, sedentary, nomadic"
    )

    def __init__(self, **data):
        # Set artifact_name before calling super to ensure filename is correct
        if "artifact_name" not in data and "common_name" in data:
            data["artifact_name"] = sanitize_for_filename(data["common_name"])
        super().__init__(**data)

    def flatten(self) -> Dict:
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}
        return {
            **parent_data,
            "species": self.common_name,
            "scientific_name": self.scientific_name,
            "extinction_year": self.extinction_year,
            "primary_cause": self.primary_cause.value,
            "habitat": self.habitat,
            "population_trajectory": [
                p.model_dump() for p in self.population_trajectory
            ],
            "cascade_effects": self.cascade_effects,
            "anthropogenic_factors": self.anthropogenic_factors,
            "symbolic_resonance": self.symbolic_resonance,
            "narrative_score": self.narrative_potential_score,
            "symbolic_weight": self.symbolic_weight,
        }

    def summary_text(self) -> str:
        """Brief summary for narrative generation"""
        pop_text = ""
        if self.population_trajectory:
            first = self.population_trajectory[0]
            last = self.population_trajectory[-1]
            pop_text = f"Population declined from {first.population} ({first.year}) to {last.population} ({last.year}). "

        return (
            f"{self.common_name} ({self.scientific_name}) went extinct in {self.extinction_year}. "
            f"{pop_text}"
            f"Primary cause: {self.primary_cause.value.replace('_', ' ')}. "
            f"Habitat: {self.habitat}. "
            f"Cascade effects: {', '.join(self.cascade_effects[:3])}."
        )

    def save_file(self):
        file = Path(self.file_path, self.file_name)
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, "w") as f:
            yaml.dump(
                self.model_dump(mode="python"),
                f,
                default_flow_style=False,
                allow_unicode=True,
            )

    def for_prompt(self):
        # ToDo: Ask claude for level of detail
        pass


if __name__ == "__main__":
    with open(
        os.path.join(
            os.getenv("AGENT_MOCK_DATA_PATH"),
            "species_extinction_artifact_mock.yml",
        ),
        "r",
    ) as file:
        data = yaml.safe_load(file)
        data["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
        extinction_artifact = SpeciesExtinctionArtifact(**data)
        print(extinction_artifact)
        extinction_artifact.save_file()
        print(extinction_artifact.flatten())
        p = extinction_artifact.for_prompt()
        print(p)
