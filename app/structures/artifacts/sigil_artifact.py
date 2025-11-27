from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import Field

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType
from app.structures.enums.sigil_state import SigilState
from app.structures.enums.sigil_type import SigilType


class SigilArtifact(ChainArtifact):
    """Record of a created sigil for the Black Agent's paranoid tracking"""

    chain_artifact_type: ChainArtifactType = ChainArtifactType.SIGIL
    chain_artifact_file_type: ChainArtifactFileType = ChainArtifactFileType.YML
    wish: Optional[str] = Field(default=None, description="Wish for the sigil.")
    statement_of_intent: Optional[str] = Field(
        default=None, description="Statement of intent for the sigil."
    )
    sigil_type: Optional[SigilType] = Field(
        default=None, description="Type of the sigil."
    )
    glyph_description: Optional[str] = Field(
        default=None, description="Description of the sigil's glyph."
    )
    glyph_components: List[str] = Field(default_factory=list)
    activation_state: Optional[SigilState] = Field(
        default=None, description="Activation state of the sigil."
    )
    charging_instructions: Optional[str] = Field(
        default=None, description="Instructions for charging the sigil."
    )

    def __init__(self, **data):
        super().__init__(**data)

    def save_file(self):
        file = Path(self.file_path, self.file_name)
        file.parent.mkdir(parents=True, exist_ok=True)
        file = Path(self.file_path, self.file_name)
        data_to_save = {
            "wish": self.wish,
            "statement_of_intent": self.statement_of_intent,
            "sigil_type": self.sigil_type.value if self.sigil_type else None,
            "glyph_description": self.glyph_description,
            "glyph_components": self.glyph_components,
            "activation_state": (
                self.activation_state.value if self.activation_state else None
            ),
            "charging_instructions": self.charging_instructions,
        }
        with open(file, "w") as f:
            yaml.dump(data_to_save, f, default_flow_style=False, allow_unicode=True)

    def flatten(self):
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}
        return {
            **parent_data,
            "thread_id": self.thread_id,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "chain_artifact_type": self.chain_artifact_type,
            "wish": self.wish,
            "statement_of_intent": self.statement_of_intent,
            "sigil_type": self.sigil_type.value if self.sigil_type else None,
            "glyph_description": self.glyph_description,
            "glyph_components": self.glyph_components,
            "activation_state": (
                self.activation_state.value if self.activation_state else None
            ),
            "charging_instructions": self.charging_instructions,
        }


if __name__ == "__main__":
    sigil = SigilArtifact(
        thread_id="test_thread_id",
        chain_artifact_file_type="yml",
        chain_artifact_type="sigil",
        wish="hi",
        statement_of_intent="hi",
        sigil_type="pictorial",
        glyph_description="hi",
        glyph_components=["hi"],
        activation_state="charging",
        charging_instructions="hi",
        base_path="/Volumes/LucidNonsense/White/chain_artifacts/",
    )

    sigil.save_file()
    print(sigil.flatten())
