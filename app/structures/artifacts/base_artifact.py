import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class ChainArtifact(BaseModel):

    chain_artifact_type: Optional[str] = Field(
        description="Type of the chain artifact.",
        default=None,
        examples=["sigil", "book", "transcript"],
    )

    def __init__(self, **data):
        super().__init__(**data)

    @staticmethod
    def save_md(
        base_path: str | Path, artifact_name: str | None, yml_block: str, jsn_block: str
    ):
        """
        Save a combined Markdown file containing YAML and JSON code blocks.
        """
        base = Path(base_path) if base_path else Path.cwd()
        base.mkdir(parents=True, exist_ok=True)
        name = artifact_name or "artifact"
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%SZ")
        out_path = base / f"{name}_{timestamp}.md"
        content = f"# Artifact: {name}\n\n{yml_block}\n\n{jsn_block}\n"
        with out_path.open("w", encoding="utf-8") as f:
            f.write(content)
        return str(out_path)
