import os
import yaml

from abc import ABC
from pathlib import Path
from typing import Optional
from pydantic import Field
from dotenv import load_dotenv

from app.structures.artifacts.html_artifact_file import HtmlChainArtifactFile
from app.structures.artifacts.template_renderer import (
    get_template_path,
    HTMLTemplateRenderer,
)
from app.structures.enums.quantum_tape_recording_quality import (
    QuantumTapeRecordingQuality,
)

load_dotenv()


class QuantumTapeLabelArtifact(HtmlChainArtifactFile, ABC):
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

    # Template fields — must match quantum_tape.html variable names exactly
    year_documented: Optional[str] = Field(
        default=None, description="Year the tape was archived (e.g. '2003')"
    )
    original_date: Optional[str] = Field(
        default=None, description="A-side real-timeline label date"
    )
    original_title: Optional[str] = Field(
        default=None, description="A-side real-timeline label text"
    )
    tapeover_date: Optional[str] = Field(
        default=None, description="B-side alternate-timeline date range"
    )
    tapeover_title: Optional[str] = Field(
        default=None, description="B-side alternate-timeline title"
    )
    subject_name: Optional[str] = Field(
        default=None, description="Biographical subject name"
    )
    age_during: Optional[str] = Field(
        default=None, description="Subject age range during the period (e.g. '22–24')"
    )
    location: Optional[str] = Field(
        default=None, description="Geographic location during the period"
    )
    catalog_number: Optional[str] = Field(
        default=None, description="Unique tape catalog identifier"
    )

    def __init__(self, **data):
        if data.get("artifact_name") in (None, "UNKNOWN_ARTIFACT_NAME") and isinstance(
            data.get("title"), str
        ):
            from app.util.string_utils import sanitize_for_filename

            data["artifact_name"] = sanitize_for_filename(data["title"])
        super().__init__(**data)

    def flatten(self):
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}
        return {
            **parent_data,
            "year_documented": self.year_documented,
            "original_date": self.original_date,
            "original_title": self.original_title,
            "tapeover_date": self.tapeover_date,
            "tapeover_title": self.tapeover_title,
            "subject_name": self.subject_name,
            "age_during": self.age_during,
            "location": self.location,
            "catalog_number": self.catalog_number,
        }

    def save_file(self):
        """Render and save the HTML file."""
        template_path = get_template_path("quantum_tape")
        renderer = HTMLTemplateRenderer(template_path)

        html_content = renderer.render_with_model(self)

        file_path = Path(self.file_path)
        file_path.mkdir(parents=True, exist_ok=True)

        output_file = file_path / self.file_name
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

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
        base_path = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
        data["base_path"] = base_path
        data["image_path"] = f"{base_path}/img"
        label = QuantumTapeLabelArtifact(**data)
        print(label)
        label.save_file()
        print(label.flatten())
        p = label.for_prompt()
        print(p)
