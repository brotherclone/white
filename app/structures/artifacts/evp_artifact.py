import logging
import os
import yaml

from abc import ABC
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field

from app.structures.artifacts.audio_artifact_file import AudioChainArtifactFile
from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType

load_dotenv()

logger = logging.getLogger(__name__)


class EVPArtifact(ChainArtifact, ABC):

    chain_artifact_type: ChainArtifactType = Field(
        default=ChainArtifactType.EVP_ARTIFACT,
        description="Type of the chain artifact should always be EVP_ARTIFACT",
    )
    chain_artifact_file_type: ChainArtifactFileType = Field(
        default=ChainArtifactFileType.YML,
        description="File format of the artifact: YAML",
    )
    transcript: Optional[str] = Field(
        default=None, description="Transcript of the EVP audio."
    )
    audio_mosiac: Optional[AudioChainArtifactFile] = Field(
        default=None, description="Audio mosaic of the EVP audio."
    )

    def __init__(self, **data):
        # Tolerate legacy fields from old YAML files
        data.pop("audio_segments", None)
        data.pop("noise_blended_audio", None)
        super().__init__(**data)

    def save_file(self):
        if self.audio_mosiac is not None:
            logger.info(f"Saving mosaic to: {self.audio_mosiac.base_path}")
            self.audio_mosiac.save_file()
        if self.base_path is None or self.thread_id not in str(self.base_path):
            self.base_path = os.path.join(
                os.getenv("AGENT_WORK_PRODUCT_BASE_PATH", "artifacts"), self.thread_id
            )
        file_path_obj = Path(self.base_path) / "yml"
        file_path_obj.mkdir(parents=True, exist_ok=True)
        file_obj = file_path_obj / self.file_name
        data_to_save = {
            "transcript": self.transcript,
            "audio_mosiac": (
                self.audio_mosiac.get_artifact_path(with_file_name=True)
                if self.audio_mosiac
                else None
            ),
        }
        with open(file_obj, "w") as f:
            yaml.dump(data_to_save, f, default_flow_style=False, allow_unicode=True)

    def flatten(self):
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}
        return {
            **parent_data,
            "chain_artifact_type": ChainArtifactType.EVP_ARTIFACT.value,
            "chain_artifact_file_type": "yaml",
            "transcript": self.transcript,
            "audio_mosiac": (
                self.audio_mosiac.get_artifact_path(with_file_name=True)
                if self.audio_mosiac
                else None
            ),
        }

    def for_prompt(self):
        prompt_parts = []
        if self.transcript:
            prompt_parts.append(f"EVP Transcript: {self.transcript}")
        if self.audio_mosiac:
            prompt_parts.append(
                f"EVP Audio Mosaic: {self.audio_mosiac.get_artifact_path(with_file_name=True)}"
            )
        return "\n".join(prompt_parts)


if __name__ == "__main__":
    with open("/Volumes/LucidNonsense/White/tests/mocks/mock.wav", "rb") as f:
        audio_bytes = f.read()
    with open(
        f"{os.getenv('AGENT_MOCK_DATA_PATH')}/black_evp_artifact_mock.yml", "r"
    ) as f:
        data = yaml.safe_load(f)
        data["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
        evp_artifact = EVPArtifact(**data)
        evp_artifact.audio_mosiac = AudioChainArtifactFile(
            thread_id="test_thread_id",
            chain_artifact_type="unknown",
            chain_artifact_file_type="wav",
            base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
            artifact_name="test_audio_artifact",
            sample_rate=44100,
            duration=5.0,
            audio_bytes=audio_bytes,
            channels=2,
        )
        evp_artifact.save_file()
        print(evp_artifact.flatten())
        p = evp_artifact.for_prompt()
        print(p)
