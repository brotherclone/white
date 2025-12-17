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

    # Primary label
    title: str  # "Summer in Portland - 1998"
    date_range: str  # "1998-06 to 1999-11"

    # Technical specs
    recording_quality: QuantumTapeRecordingQuality
    counter_start: int = Field(ge=0, le=9999)
    counter_end: Optional[int] = Field(None, ge=0, le=9999)

    # Handwritten notes
    notes: Optional[str] = None  # "Wrote novel. Didn't show anyone."

    # Analog artifacts
    original_label_visible: bool = True  # Is the original label still visible?
    original_label_text: Optional[str] = None
    tape_degradation: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Visual metadata
    tape_brand: Optional[str] = Field(default="TASCAM 424-S")
    handwriting_style: Optional[str] = None

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
