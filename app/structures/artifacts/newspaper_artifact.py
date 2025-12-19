import os
import yaml

from abc import ABC
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType
from app.util.string_utils import sanitize_for_filename

load_dotenv()


class NewspaperArtifact(ChainArtifact, ABC):

    chain_artifact_type: ChainArtifactType = ChainArtifactType.NEWSPAPER_ARTICLE
    headline: Optional[str] = Field(
        default=None, description="Headline of the newspaper article."
    )
    date: Optional[str] = Field(
        default=None, description="Publication date of the article."
    )
    source: Optional[str] = Field(
        default=None, description="Source of the newspaper article."
    )
    location: Optional[str] = Field(
        default=None, description="Location related to the article."
    )
    text: Optional[str] = Field(
        default=None, description="Full text of the newspaper article."
    )
    tags: Optional[list[str]] = Field(
        default=None, description="Tags associated with the article."
    )

    def __init__(self, **data):
        # Set artifact_name before calling super to ensure filename is correct
        if "artifact_name" not in data and "headline" in data and data["headline"]:
            data["artifact_name"] = sanitize_for_filename(data["headline"])
        super().__init__(**data)

    def get_text_content(self) -> str:
        """
        Get the full text content of the article.
        Returns the text field as-is. Used for compatibility with agent processing.
        """
        return self.text if self.text else ""

    def to_markdown(self) -> str:
        """Convert newspaper artifact to formatted markdown."""
        md_lines = []

        # Headline
        if self.headline:
            md_lines.append(f"# {self.headline}")
            md_lines.append("")

        # Metadata
        if self.source:
            md_lines.append(f"**Source:** {self.source}")
        if self.date:
            md_lines.append(f"**Date:** {self.date}")
        if self.location:
            md_lines.append(f"**Location:** {self.location}")
        md_lines.append("")

        # Article text
        if self.text:
            md_lines.append(self.text)
            md_lines.append("")

        # Tags
        if self.tags:
            md_lines.append(f"**Tags:** {', '.join(self.tags)}")
            md_lines.append("")

        # Metadata footer
        md_lines.append("---")
        md_lines.append("## Metadata")
        md_lines.append(f"- **Artifact ID:** {self.artifact_id}")
        md_lines.append(f"- **Thread ID:** {self.thread_id}")
        md_lines.append(
            f"- **Rainbow Color:** {self.rainbow_color_mnemonic_character_value}"
        )

        return "\n".join(md_lines)

    def save_file(self):
        file = Path(self.file_path, self.file_name)
        file.parent.mkdir(parents=True, exist_ok=True)
        file = Path(self.file_path, self.file_name)
        with open(file, "w") as f:
            if self.chain_artifact_file_type == ChainArtifactFileType.MARKDOWN:
                f.write(self.to_markdown())
            else:
                yaml.dump(
                    self.model_dump(mode="python"),
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                )

    def flatten(self):
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}
        return {
            **parent_data,
            "thread_id": self.thread_id,
            "chain_artifact_file_type": self.chain_artifact_file_type.value,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "chain_artifact_type": ChainArtifactType.NEWSPAPER_ARTICLE.value,
            "headline": self.headline,
            "date": self.date,
            "source": self.source,
            "location": self.location,
            "text": self.text,
            "tags": self.tags,
        }

    def for_prompt(self) -> str:
        """Format for prompt - newspaper article with context."""
        parts = []
        if self.headline:
            parts.append(self.headline)
        metadata_parts = []
        if self.source:
            metadata_parts.append(self.source)
        if self.date:
            metadata_parts.append(self.date)
        if self.location:
            metadata_parts.append(self.location)
        if metadata_parts:
            parts.append(" | ".join(metadata_parts))
        parts.append("")
        if self.text:
            parts.append(self.text)
        if self.tags:
            parts.append(f"\n[{', '.join(self.tags)}]")

        return "\n".join(parts)


if __name__ == "__main__":
    with open(
        os.path.join(os.getenv("AGENT_MOCK_DATA_PATH"), "orange_base_story_mock.yml"),
        "r",
    ) as f:
        data = yaml.safe_load(f)
        newspaper_artifact = NewspaperArtifact(**data)
    newspaper_artifact.save_file()
    print(newspaper_artifact.flatten())
    p = newspaper_artifact.for_prompt()
    print(p)
