import errno
import os
import tempfile
import yaml

from dotenv import load_dotenv
from abc import ABC
from pathlib import Path
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
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType
from app.util.string_utils import sanitize_for_filename

load_dotenv()


class LastHumanSpeciesExtinctionNarrativeArtifact(ChainArtifact, ABC):
    """
    Interleaved narrative showing species and human collapse in parallel.
    Structure: documentary-intimate duet.
    """

    chain_artifact_type: ChainArtifactType = (
        ChainArtifactType.LAST_HUMAN_SPECIES_EXTINCTION_NARRATIVE
    )
    chain_artifact_file_type: ChainArtifactFileType = ChainArtifactFileType.MARKDOWN
    rainbow_color_mnemonic_character_value: str = "G"

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

    def __init__(self, **data):
        # Set artifact_name before calling super to ensure filename is correct
        if "artifact_name" not in data and "species" in data and "human" in data:
            species = data["species"]
            human = data["human"]
            species_name = (
                species.common_name
                if hasattr(species, "common_name")
                else str(species.get("common_name", "unknown"))
            )
            human_name = (
                human.name
                if hasattr(human, "name")
                else str(human.get("name", "unknown"))
            )
            data["artifact_name"] = sanitize_for_filename(
                f"{species_name}_{human_name}_narrative"
            )
        super().__init__(**data)

    def flatten(self) -> Dict:
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}
        return {
            **parent_data,
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
        """Format as readable Markdown for artifact storage"""
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

    def save_file(self):
        """
        Persist the artifact as Markdown (.md). Ensures parent directories exist,
        forces a .md suffix, and falls back to the system temp dir if the target
        directory is not writable (read-only filesystem).
        """
        base_path = Path(self.file_path or ".").expanduser()
        file_name = (self.file_name or "artifact.md").strip()

        file_path = base_path / file_name
        if file_path.suffix.lower() != ".md":
            file_path = file_path.with_suffix(".md")

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # errno.EROFS == 30 on macOS/Linux -> read-only filesystem
            if getattr(e, "errno", None) == errno.EROFS:
                fallback_dir = Path(tempfile.gettempdir()).expanduser()
                fallback_path = fallback_dir / file_path.name
                fallback_path.parent.mkdir(parents=True, exist_ok=True)
                file_path = fallback_path
                print(
                    f"Warning: target directory not writable. Falling back to `{file_path}`"
                )
            else:
                raise

        with file_path.open("w", encoding="utf-8") as f:
            f.write(self.to_markdown())


if __name__ == "__main__":
    with open(
        os.path.join(
            os.getenv("AGENT_MOCK_DATA_PATH"),
            "last_human_species_extinction_narrative.yml",
        ),
        "r",
    ) as f:
        data = yaml.safe_load(f)
        data["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
        last_human_narrative_artifact = LastHumanSpeciesExtinctionNarrativeArtifact(
            **data
        )
        print(last_human_narrative_artifact)
        last_human_narrative_artifact.save_file()
        print(last_human_narrative_artifact.flatten())
