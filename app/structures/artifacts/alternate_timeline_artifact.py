import os
import yaml
from abc import ABC
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import Field, field_validator

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.concepts.alternate_life_detail import AlternateLifeDetail
from app.structures.concepts.biographical_period import BiographicalPeriod
from app.structures.concepts.divergence_point import DivergencePoint
from app.structures.enums.quantum_tape_emotional_tone import QuantumTapeEmotionalTone

load_dotenv()


class AlternateTimelineArtifact(ChainArtifact, ABC):
    """The fictional-but-plausible alternate life."""

    period: BiographicalPeriod = Field(
        alias="period", description="Biographical period of the alternate timeline."
    )
    title: str = Field(description="Title of the alternate timeline.")
    narrative: str = Field(
        description="Narrative of the alternate timeline.", min_length=100
    )
    divergence_point: DivergencePoint = Field(
        alias="divergence_point",
        description="Point of divergence between the original timeline and the alternate timeline.",
    )
    key_differences: List[str] = Field(
        description="Key differences in the alternate timeline."
    )
    specific_details: List[AlternateLifeDetail] = Field(
        description="Specific details of the alternate timeline."
    )
    emotional_tone: QuantumTapeEmotionalTone = Field(
        alias="emotional_tone", description="Emotional tone of the alternate timeline."
    )
    mood_description: str = Field(
        description="Phenomenological notes about the alternate timeline."
    )
    preceding_events: List[str] = Field(
        description="Events that happened before the alternate timeline."
    )
    following_events: List[str] = Field(
        description="Events that happened after the alternate timeline."
    )
    plausibility_score: Optional[float] = Field(
        default=None, description="How plausible the divergent event is", ge=0.0, le=1.0
    )
    specificity_score: Optional[float] = Field(
        default=None,
        description="How specific the narrative details are",
        ge=0.0,
        le=1.0,
    )
    divergence_magnitude: Optional[float] = Field(
        default=None,
        description="How disruptive this event will be to other timelines",
        ge=0.0,
        le=1.0,
    )

    def __init__(self, **data):
        super().__init__(**data)

    @field_validator("narrative")
    @classmethod
    def validate_narrative_length(cls, v):
        """Narrative must be substantive."""
        if len(v.split()) < 100:
            raise ValueError("Narrative too short (minimum 100 words)")
        return v

    @field_validator("specific_details")
    @classmethod
    def validate_enough_details(cls, v):
        """Must have at least 5 specific details."""
        if len(v) < 5:
            raise ValueError(f"Not enough specific details ({len(v)}/5)")
        return v

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

    @staticmethod
    def _format_items(
        items: List[str] | List[AlternateLifeDetail], bullet: str, indent: int = 3
    ) -> str:
        """Return a consistently indented, newline-separated block of bullets.
        - Empty lists return an empty string.
        - AlternateLifeDetail items attempt `model_dump()` and fall back to `str()`."""
        if not items:
            return ""
        prefix = " " * indent

        def render(item):
            if isinstance(item, AlternateLifeDetail):
                item_data = item.model_dump()
                return (
                    item_data.get("text")
                    or item_data.get("description")
                    or item_data.get("detail")
                    or (
                        ", ".join(f"{k}: {v}" for k, v in item_data.items())
                        if isinstance(item_data, dict)
                        else str(item_data)
                    )
                )
            return str(item)

        return "\n".join(f"{prefix}{bullet} {render(it)}" for it in items)

    def for_prompt(self):
        preceding_block = self._format_items(self.preceding_events, "⍿", indent=3)
        key_diff_block = self._format_items(self.key_differences, "≏", indent=8)
        specific_block = self._format_items(self.specific_details, "✧", indent=8)
        following_block = self._format_items(self.following_events, "⍿", indent=3)

        artifact_for_prompt = (
            f"Alternate Timeline:\n"
            f"Title: {self.title} | {self.period.start_date} - {self.period.end_date} | Age: {self.period.age_range}\n\n"
            f"{self.narrative}\n\n"
            f"{preceding_block}\n"
            f"⌇ Timeline Anomaly Level {self.divergence_magnitude}: {self.divergence_point.when}: {self.divergence_point.what_changed}\n"
            f"    Key Differences:\n{key_diff_block}\n"
            f"    Specific Details:\n{specific_block}\n"
            f"    Emotional Tone: {self.emotional_tone.value}\n"
            f"    Phenomenological Notes: {self.mood_description}\n"
            f"{following_block}\n"
        )
        return artifact_for_prompt


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
        p = timeline.for_prompt()
        print(p)
