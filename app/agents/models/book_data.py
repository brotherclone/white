from typing import Optional, List, Literal

from pydantic import BaseModel, Field

from app.agents.enums.book_condition import BookCondition
from app.agents.enums.publisher_type import PublisherType


class BookData(BaseModel):

    """Record of a created book for the Red Agent's book tracking"""

    title: str = Field(..., description="Full title of the work")
    subtitle: Optional[str] = Field(None, description="Subtitle if present")
    author: str = Field(..., description="Author or attributed author")
    author_credentials: Optional[str] = Field(None, description="Academic or occult credentials")
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
    translated_from: Optional[str] = Field(None, description="Original language if translated")
    translator: Optional[str] = Field(None, description="Translator name")
    tags: List[str] = Field(default_factory=list, description="Subject tags")
    danger_level: int = Field(..., description="1=curious, 5=forbidden")
    abstract: Optional[str] = Field(None, description="Brief description")
    notable_quote: Optional[str] = Field(None, description="Memorable excerpt")
    suppression_history: Optional[str] = Field(None, description="Censorship notes")
    related_works: List[str] = Field(default_factory=list, description="Related titles")