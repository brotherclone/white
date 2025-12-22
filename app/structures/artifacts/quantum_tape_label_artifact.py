import os
import yaml

from abc import ABC
from pathlib import Path
from typing import Optional
from pydantic import Field
from dotenv import load_dotenv

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.enums.quantum_tape_recording_quality import (
    QuantumTapeRecordingQuality,
)

load_dotenv()


class QuantumTapeLabelArtifact(ChainArtifact, ABC):
    """VHS/Cassette tape label metadata."""

    title: str = Field(
        description="The title of the tape", examples=["Summer in Portland - 1998"]
    )
    date_range: str = Field(
        description="Date range for the recording", examples=["1998-06 to 1999-11"]
    )
    recording_quality: QuantumTapeRecordingQuality = Field(
        description="Quality of the recording"
    )
    counter_start: int = Field(
        default=0, ge=0, le=9999, description="Counter start value"
    )
    counter_end: Optional[int] = Field(
        default=None, ge=0, le=9999, description="Counter end value"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Biographical glimpses in handwritten notes",
        examples=["Wrote novel. Didn't show anyone."],
    )
    original_label_visible: bool = Field(
        default=True, description="Is the original label still visible?"
    )
    original_label_text: Optional[str] = Field(
        default=None, description="The original label text"
    )
    tape_degradation: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Degradation of the tape watchout Basinski",
    )
    tape_brand: Optional[str] = Field(
        default="TASCAM 424-S",
        description="Brand of the tape - metal preferred for for 4-tracking!",
    )
    handwriting_style: Optional[str] = Field(
        default=None, description="Handwriting style of the notes"
    )

    def __init__(self, **data):
        super().__init__(**data)

    def flatten(self):
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}
        return {
            **parent_data,
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

    def for_prompt(self):
        prompt_parts = [f"Title: {self.title}", f"Date Range: {self.date_range}"]
        if self.original_label_visible:
            prompt_parts.append(f"Original Label Text: {self.original_label_text}")
        return "\n".join(prompt_parts)


if __name__ == "__main__":
    with open(
        f"{os.getenv('AGENT_MOCK_DATA_PATH')}/quantum_tape_label_mock.yml", "r"
    ) as a_file:
        data = yaml.safe_load(a_file)
        data["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
        label = QuantumTapeLabelArtifact(**data)
        print(label)
        label.save_file()
        print(label.flatten())
        p = label.for_prompt()
        print(p)
