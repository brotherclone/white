from typing import Optional, Any

from pydantic import Field, field_validator

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.artifacts.text_artifact_file import TextChainArtifactFile


class NewspaperArtifact(ChainArtifact):

    thread_id: str
    chain_artifact_type: str = "newspaper_article"
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
    page: Optional[TextChainArtifactFile] = Field(
        default=None, description="Page of the newspaper article."
    )

    @field_validator("page", mode="before")
    def _normalize_page(cls, v: Any) -> Any:
        if isinstance(v, list) and len(v) > 0:
            v = v[0]
        if isinstance(v, dict):
            v.setdefault("base_path", "")
            try:
                return TextChainArtifactFile(**v)
            except ValueError:
                return v
        return v

    def get_text_content(self) -> Optional[str]:
        text_content = f"""
        {self.headline}
        {self.date} | {self.source} | {self.location}
        {self.text}
        """
        return text_content
