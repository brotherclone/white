import os
from abc import ABC

import yaml
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType
from app.structures.enums.symbolic_object_category import SymbolicObjectCategory

load_dotenv()


class SymbolicObjectArtifact(ChainArtifact, ABC):
    """Symbolic object that has emerged from the nostalgia of Rows Bud, the orange agent"""

    # ToDo: Needs a for_prompt() method

    chain_artifact_type: ChainArtifactType = ChainArtifactType.SYMBOLIC_OBJECT
    chain_artifact_file_type: ChainArtifactFileType = ChainArtifactFileType.YML
    symbolic_object_category: SymbolicObjectCategory = Field(
        description="""Category of the object: 
    CIRCULAR_TIME - Clocks, calendars, loops, temporal markers (Nash's 182 BPM clock)
    INFORMATION_ARTIFACTS - Newspapers, broadcasts, transmissions, EMORYs, recordings
    LIMINAL_OBJECTS - Doorways, thresholds, portals, Pine Barrens gateways
    PSYCHOGEOGRAPHIC - Maps, coordinates, dimensional markers, location-based objects""",
        default=SymbolicObjectCategory.INFORMATION_ARTIFACTS,
    )
    name: Optional[str] = Field(default=None, description="A name for the object.")
    description: Optional[str] = Field(
        default=None, description="A detailed description of the object."
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
            "chain_artifact_type": ChainArtifactType.SYMBOLIC_OBJECT.value,
            "symbolic_object_category": self.symbolic_object_category.value,
            "name": self.name,
            "description": self.description,
        }


if __name__ == "__main__":
    with open(
        os.path.join(
            os.getenv("AGENT_MOCK_DATA_PATH"),
            "orange_mock_object_selection.yml",
        ),
        "r",
    ) as f:
        data = yaml.safe_load(f)
        symb_obj = SymbolicObjectArtifact(**data)
        symb_obj.save_file()
        print(symb_obj.flatten())
