from pydantic import BaseModel
from pathlib import Path
from datetime import datetime


class ChainArtifact(BaseModel):

    chain_artifact_type: str | None = None

    def __init__(self, **data):
        super().__init__(**data)

    @staticmethod
    def save_md(base_path: str | Path, artifact_name: str | None, yml_block: str, jsn_block: str):
        """Save a combined markdown file containing YAML and JSON code blocks.

        This is a small helper used by tools to persist a textual representation of an artifact.
        """
        base = Path(base_path) if base_path else Path.cwd()
        base.mkdir(parents=True, exist_ok=True)
        name = artifact_name or "artifact"
        # include timestamp to avoid overwriting during tests
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_path = base / f"{name}_{timestamp}.md"
        content = f"# Artifact: {name}\n\n{yml_block}\n\n{jsn_block}\n"
        with out_path.open("w", encoding="utf-8") as f:
            f.write(content)
        return str(out_path)
