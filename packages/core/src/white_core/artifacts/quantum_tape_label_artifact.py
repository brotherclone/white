import os
from abc import ABC
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import Field

from white_core.artifacts.html_artifact_file import HtmlChainArtifactFile
from white_core.enums.quantum_tape_recording_quality import (
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
            from white_core.util.string_utils import sanitize_for_filename

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
        original_label_section = ""
        if self.original_label_visible:
            label_text = self.original_label_text or ""
            original_label_section = (
                f'<p class="orig-label">Original Label: {label_text}</p>'
            )

        notes_section = f'<p class="notes">{self.notes}</p>' if self.notes else ""

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>{self.title}</title>
  <style>
    body {{ font-family: "Courier New", Courier, monospace; background: #1a1a1a; color: #e8e0d0; margin: 0; padding: 2rem; }}
    .label {{ max-width: 640px; margin: 0 auto; border: 2px solid #6b5e3e; padding: 1.5rem 2rem; background: #111; }}
    h1 {{ font-size: 1.4rem; margin: 0 0 0.25rem; }}
    .date-range {{ color: #a09070; font-size: 0.85rem; margin-bottom: 1rem; }}
    .sides {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0; }}
    .side {{ border: 1px solid #3a3028; padding: 0.5rem; }}
    .side-label {{ font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.2em; color: #7a6a50; }}
    .side-date {{ font-size: 0.8rem; color: #a09070; margin: 0.2rem 0; }}
    .side-title {{ font-weight: bold; }}
    .meta {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem 1.5rem; margin-top: 1rem; font-size: 0.85rem; }}
    .meta-key {{ font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.15em; color: #7a6a50; }}
    .orig-label {{ margin-top: 1rem; font-style: italic; color: #8a7a60; font-size: 0.85rem; }}
    .notes {{ margin-top: 1rem; color: #a09070; font-size: 0.85rem; }}
    .catalog {{ margin-top: 1rem; text-align: right; font-size: 0.7rem; color: #6b5e3e; letter-spacing: 0.1em; }}
  </style>
</head>
<body>
  <div class="label">
    <h1>{self.title}</h1>
    <div class="date-range">{self.date_range}</div>
    <div class="sides">
      <div class="side">
        <div class="side-label">Original Timeline</div>
        <div class="side-date">{self.original_date or ""}</div>
        <div class="side-title">{self.original_title or ""}</div>
      </div>
      <div class="side">
        <div class="side-label">Tapeover</div>
        <div class="side-date">{self.tapeover_date or ""}</div>
        <div class="side-title">{self.tapeover_title or ""}</div>
      </div>
    </div>
    <div class="meta">
      <div><div class="meta-key">Subject</div>{self.subject_name or ""}</div>
      <div><div class="meta-key">Age During</div>{self.age_during or ""}</div>
      <div><div class="meta-key">Location</div>{self.location or ""}</div>
      <div><div class="meta-key">Year Documented</div>{self.year_documented or ""}</div>
    </div>
    {original_label_section}
    {notes_section}
    <div class="catalog">{self.catalog_number or ""}</div>
  </div>
</body>
</html>"""

        file_path = Path(self.file_path)
        file_path.mkdir(parents=True, exist_ok=True)

        if not self.file_name:
            raise ValueError("file_name is not set; cannot save file.")
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
        example_data = yaml.safe_load(a_file)
        base_path = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
        example_data["base_path"] = base_path
        example_data["image_path"] = f"{base_path}/img"
        label = QuantumTapeLabelArtifact(**example_data)
        print(label)
        label.save_file()
        print(label.flatten())
        p = label.for_prompt()
        print(p)
