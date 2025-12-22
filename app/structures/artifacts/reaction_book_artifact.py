from abc import ABC
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import Field

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.enums.book_condition import BookCondition
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType
from app.structures.enums.publisher_type import PublisherType


class ReactionBookChainArtifact(ChainArtifact, ABC):

    chain_artifact_type: ChainArtifactType = ChainArtifactType.BOOK
    chain_artifact_file_type: ChainArtifactFileType = ChainArtifactFileType.YML
    title: str = Field(..., description="Full title of the work")
    subtitle: Optional[str] = Field(None, description="Subtitle if present")
    author: str = Field(..., description="Author or attributed author")
    author_credentials: Optional[str] = Field(
        None, description="Academic or occult credentials"
    )
    year: int = Field(..., description="Year of publication")
    publisher: str = Field(..., description="Publisher name")
    publisher_type: PublisherType = Field(..., description="Type of publisher")
    edition: str = Field(default="1st", description="Edition information")
    pages: int = Field(..., description="Page count")
    isbn: Optional[str] = Field(None, description="ISBN if it exists")
    catalog_number: str = Field(..., description="Red Agent's catalog number")
    condition: BookCondition = Field(..., description="Physical condition")
    acquisition_date: Optional[str] = Field(None, description="When acquired")
    acquisition_notes: Optional[str] = Field(None, description="How it was acquired")
    language: str = Field(default="English", description="Primary language")
    translated_from: Optional[str] = Field(
        None, description="Original language if translated"
    )
    translator: Optional[str] = Field(None, description="Translator name")
    tags: List[str] = Field(default_factory=list, description="Subject tags")
    danger_level: int = Field(..., description="1=curious, 5=forbidden")
    abstract: Optional[str] = Field(None, description="Brief description")
    notable_quote: Optional[str] = Field(None, description="Memorable excerpt")
    suppression_history: Optional[str] = Field(None, description="Censorship notes")
    related_works: List[str] = Field(default_factory=list, description="Related titles")
    excerpts: Optional[List[str]] = Field(
        default=None,
        description="Excerpts associated with the artifact that will be generated.",
        examples=[
            "Milk perhaps is an ego wrapped in theory.",
            "The literature would have you believe a ham shell is a horse string.",
        ],
    )
    original_book_title: str
    original_book_author: str

    def __init__(self, **data):
        super().__init__(**data)

    def save_file(self):
        file = Path(self.file_path, self.file_name)
        file.parent.mkdir(parents=True, exist_ok=True)
        file = Path(self.file_path, self.file_name)
        with open(file, "w") as f:
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
            "original_book_title": self.original_book_title,
            "original_book_author": self.original_book_author,
            "thread_id": self.thread_id,
            "chain_artifact_file_type": ChainArtifactFileType.YML.value,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "chain_artifact_type": ChainArtifactType.BOOK.value,
            "title": self.title,
            "subtitle": self.subtitle,
            "author": self.author,
            "author_credentials": self.author_credentials,
            "year": self.year,
            "publisher": self.publisher,
            "publisher_type": self.publisher_type.value,
            "edition": self.edition,
            "pages": self.pages,
            "isbn": self.isbn,
            "catalog_number": self.catalog_number,
            "condition": self.condition.value,
            "acquisition_date": self.acquisition_date,
            "acquisition_notes": self.acquisition_notes,
            "language": self.language,
            "translated_from": self.translated_from,
            "translator": self.translator,
            "tags": self.tags,
            "danger_level": self.danger_level,
            "abstract": self.abstract,
            "notable_quote": self.notable_quote,
            "suppression_history": self.suppression_history,
            "related_works": self.related_works,
            "excerpts": self.excerpts,
        }

    def for_prompt(self) -> str:
        """Return a human-readable summary for prompting.

        Includes title, subtitle (if present), author, year, publisher,
        condition, danger level and the generated file path/name.
        """
        subtitle_part = f" — {self.subtitle}" if self.subtitle else ""
        condition = getattr(self.condition, "value", str(self.condition))
        file_info = f"{self.get_artifact_path(with_file_name=True)}"

        parts = [
            f"Book: {self.title}{subtitle_part}",
            f"Author: {self.author}",
            f"Year: {self.year}",
            f"Publisher: {self.publisher} ({getattr(self.publisher_type, 'value', self.publisher_type)})",
            f"Condition: {condition}",
            f"Danger level: {self.danger_level}",
            f"File: {file_info}",
        ]

        if self.abstract:
            parts.append(f"Abstract: {self.abstract}")
        if self.notable_quote:
            parts.append(f"Notable quote: {self.notable_quote}")

        return ", ".join(parts)


if __name__ == "__main__":

    book_artifact = ReactionBookChainArtifact(
        original_book_title="Ham Shell",
        original_book_author="Ham Shell Press",
        thread_id="test_thread_id",
        chain_artifact_type="book",
        title="The Ham Shell",
        subtitle="A Tale of Ham and Spam",
        author="",
        base_path="/Volumes/LucidNonsense/White/chain_artifacts/",
        publisher="The Ham Shell Press",
        publisher_type="Samizdat dead drop",
        edition="1st",
        pages=100,
        danger_level=3,
        isbn="978-0-19-853222-1",
        catalog_number="123456789",
        year=2023,
        condition="good",
        language="English",
        translated_from="French",
        translator="Jane Doe",
        tags=["ham", "spam"],
        abstract="This is a book about ham and spam.",
        notable_quote="Ham is a ham shell.",
        suppression_history="Censored",
        related_works=["Ham Shell 2"],
        excerpts=["Ham shells are ham.", "Spam is spam."],
    )
    book_artifact.save_file()
    print(book_artifact.flatten())
