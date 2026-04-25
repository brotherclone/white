import logging
import os
import tempfile
from abc import ABC
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import Field, field_validator

from white_core.artifacts.base_artifact import ChainArtifact
from white_core.artifacts.book_artifact import BookArtifact
from white_core.enums.chain_artifact_file_type import ChainArtifactFileType
from white_core.enums.chain_artifact_type import ChainArtifactType

load_dotenv()

logger = logging.getLogger(__name__)


class CardCatalogArtifact(ChainArtifact, ABC):
    """Card Catalog artifact for documenting forbidden/dangerous books."""

    chain_artifact_type: ChainArtifactType = Field(
        default=ChainArtifactType.CARD_CATALOG,
        description="Type: Card Catalog Entry",
    )
    chain_artifact_file_type: ChainArtifactFileType = Field(
        default=ChainArtifactFileType.YML, description="File type: YML"
    )
    artifact_name: str = Field(default="card_catalog", description="Artifact name")

    danger_level: int = Field(default=1, description="Danger level 1-5", ge=1, le=5)
    acquisition_date: str = Field(description="Date acquired")
    title: str = Field(description="Book title")
    subtitle: str = Field(default="", description="Book subtitle")
    author: str = Field(description="Author name")
    author_credentials: Optional[str] = Field(
        default=None, description="Author credentials"
    )
    year: str = Field(description="Publication year")
    publisher: str = Field(description="Publisher name")
    publisher_type: str = Field(default="Academic Press", description="Publisher type")
    edition: str = Field(default="First Edition", description="Edition info")
    pages: int = Field(description="Number of pages")
    isbn: Optional[str] = Field(default=None, description="ISBN")
    language: str = Field(default="English", description="Language")
    translated_from: Optional[str] = Field(
        default=None, description="Original language if translated"
    )
    condition: str = Field(default="Good", description="Physical condition")
    abstract: str = Field(description="Book abstract")
    notable_quote: Optional[str] = Field(
        default=None, description="Notable quote from the book"
    )
    suppression_history: Optional[str] = Field(
        default=None, description="Suppression history"
    )
    tags: List[str] = Field(default_factory=list, description="Category tags")
    acquisition_notes: Optional[str] = Field(
        default=None, description="How it was acquired"
    )
    catalog_number: str = Field(description="Catalog reference number")

    @field_validator("year", mode="before")
    @classmethod
    def _convert_year_to_str(cls, v):
        if v is None:
            return v
        return str(v)

    def save_file(self):
        try:
            file_path = Path(self.file_path)
            file_path.mkdir(parents=True, exist_ok=True)
        except (FileNotFoundError, OSError) as e:
            logger.warning(
                f"Could not create directory `{self.file_path}`: {e!s}. Falling back to system temp directory."
            )
            fallback_dir = Path(tempfile.gettempdir()) / "agent_work_product"
            try:
                fallback_dir.mkdir(parents=True, exist_ok=True)
                file_path = fallback_dir
            except Exception as fallback_err:
                logger.error(
                    f"Failed to create fallback directory `{fallback_dir}`: {fallback_err!s}. Using current working directory."
                )
                file_path = Path.cwd()

        output_file = file_path / self.file_name
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.model_dump(mode="json"),
                f,
                default_flow_style=False,
                allow_unicode=True,
                width=float("inf"),
            )
        logger.info(f"Wrote artifact to `{output_file}`")

    def flatten(self):
        return self.model_dump(mode="json")

    def for_prompt(self) -> str:
        return f"""Card Catalog Entry: {self.catalog_number}
Title: {self.title}
Author: {self.author}
Published: {self.year} by {self.publisher}
Danger Level: {self.danger_level}/5
Abstract: {self.abstract}
Tags: {', '.join(self.tags)}"""


if __name__ == "__main__":
    with open(
        f"{os.getenv('AGENT_MOCK_DATA_PATH')}/red_book_artifact_mock.yml", "r"
    ) as f:
        data = yaml.safe_load(f)
        data["thread_id"] = "mock_thread_001"
        book_artifact = BookArtifact(**data)

    base_path = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
    book_data = book_artifact.flatten()
    book_data.pop("chain_artifact_file_type", None)
    book_data.pop("file_name", None)
    book_data.pop("file_path", None)

    c = CardCatalogArtifact(**book_data, base_path=base_path)
    print(c.for_prompt())
    c.save_file()
