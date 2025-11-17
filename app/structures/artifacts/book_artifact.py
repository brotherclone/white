import os
import yaml

from dotenv import load_dotenv
from abc import ABC
from typing import List, Optional
from pydantic import Field, BaseModel
from pathlib import Path

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.enums.book_condition import BookCondition
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType
from app.structures.enums.publisher_type import PublisherType

load_dotenv()


class BookDataPageCollection(BaseModel):

    page_1_text: str = Field(..., description="Page 1 of the book")
    page_2_text: str = Field(..., description="Page 2 of the book")

    def __init__(self, **data):
        super().__init__(**data)


class BookArtifact(ChainArtifact, ABC):
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


if __name__ == "__main__":

    with open(
        f"{os.getenv('AGENT_MOCK_DATA_PATH')}/red_book_artifact_mock.yml", "r"
    ) as f:
        data = yaml.safe_load(f)
        book_artifact = BookArtifact(**data)
        book_artifact.save_file()
        print(book_artifact.flatten())
