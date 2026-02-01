import os
import numpy as np
from dotenv import load_dotenv
from abc import ABC
from pydantic import Field
from scipy.io import wavfile
from pathlib import Path

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType

load_dotenv()


class AudioChainArtifactFile(ChainArtifact, ABC):
    chain_artifact_file_type: ChainArtifactFileType = Field(
        default=ChainArtifactFileType.AUDIO,
        description="Type of the chain artifact file should always be audio (.wav) in this case..",
    )
    sample_rate: int = Field(
        description="Sample rate of the audio file", ge=24000, le=96000, default=44100
    )
    duration: float = Field(
        description="Duration of the audio file in seconds",
        ge=0.0,
        le=1000000.0,
        default=1.0,
    )
    audio_bytes: bytes = Field(default=b"", description="Audio bytes")
    channels: int = Field(description="Number of audio channels", ge=1, le=2, default=2)

    def __init__(self, **data):
        super().__init__(**data)

    def save_file(self):
        # Use the computed file_path property from base class
        # This already includes base_path/thread_id/wav
        file_path_obj = Path(self.file_path)
        file_path_obj.mkdir(parents=True, exist_ok=True)
        file_path = file_path_obj / self.file_name

        audio_array = np.frombuffer(self.audio_bytes, dtype=np.int16)
        wavfile.write(str(file_path), self.sample_rate, audio_array)

    def flatten(self):
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}

        return {
            **parent_data,
            "thread_id": self.thread_id,
            "chain_artifact_type": self.chain_artifact_type.value,
            "chain_artifact_file_type": ChainArtifactFileType.AUDIO.value,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "channels": self.channels,
        }

    def for_prompt(self):
        return (
            f"Audio file: {self.get_artifact_path(True)}, {self.duration} seconds long."
        )


if __name__ == "__main__":
    with open("/Volumes/LucidNonsense/White/tests/mocks/mock.wav", "rb") as f:
        audio_bytes = f.read()

    audio_artifact = AudioChainArtifactFile(
        thread_id="test_thread_id",
        base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
        chain_artifact_type="unknown",
        chain_artifact_file_type="wav",
        sample_rate=44100,
        duration=5.0,
        audio_bytes=audio_bytes,
        channels=2,
    )

    audio_artifact.save_file()
    print(audio_artifact.flatten())
    p = audio_artifact.for_prompt()
    print(p)
