import os
import yaml
from abc import ABC
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import Field

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.concepts.alternate_life_detail import AlternateLifeDetail
from app.structures.concepts.biographical_period import BiographicalPeriod
from app.structures.concepts.divergence_point import DivergencePoint
from app.structures.enums.quantum_tape_emotional_tone import QuantumTapeEmotionalTone

load_dotenv()


class AlternateTimelineArtifact(ChainArtifact, ABC):
    """The fictional-but-plausible alternate life."""

    # Core narrative
    period: BiographicalPeriod
    title: str  # "Summer in Portland, 1998"
    narrative: str  # Full prose description

    # Structure
    divergence_point: DivergencePoint
    key_differences: List[str]  # Bullet points of what changed
    specific_details: List[AlternateLifeDetail]

    # Tone
    emotional_tone: QuantumTapeEmotionalTone
    mood_description: str

    # Context
    preceding_events: List[str]  # What happened before in actual timeline
    following_events: List[str]  # What happened after in actual timeline

    # Quality metrics
    plausibility_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    specificity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    divergence_magnitude: Optional[float] = Field(None, ge=0.0, le=1.0)

    def __init__(self, **data):
        super().__init__(**data)

    def flatten(self):
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}
        return {
            **parent_data,
            "period": self.period.model_dump(),
            "title": self.title,
            "narrative": self.narrative,
            "divergence_point": self.divergence_point.model_dump(),
            "key_differences": self.key_differences,
            "specific_details": [
                detail.model_dump() for detail in self.specific_details
            ],
            "emotional_tone": self.emotional_tone.value,
            "mood_description": self.mood_description,
            "preceding_events": self.preceding_events,
            "following_events": self.following_events,
            "plausibility_score": self.plausibility_score,
            "specificity_score": self.specificity_score,
            "divergence_magnitude": self.divergence_magnitude,
        }

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


if __name__ == "__main__":
    with open(
        f"{os.getenv('AGENT_MOCK_DATA_PATH')}/alternate_timeline_artifact_mock.yml", "r"
    ) as a_file:
        data = yaml.safe_load(a_file)
        data["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
        timeline = AlternateTimelineArtifact(**data)
        print(timeline)
        timeline.save_file()
        print(timeline.flatten())
