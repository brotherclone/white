import logging
import os
import yaml

from pathlib import Path
from typing import List
from dotenv import load_dotenv

from app.structures.artifacts.audio_artifact_file import AudioChainArtifactFile
from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType

load_dotenv()


class EVPArtifact(ChainArtifact):

    chain_artifact_type: ChainArtifactType = ChainArtifactType.EVP_ARTIFACT
    chain_artifact_file_type: ChainArtifactFileType = ChainArtifactFileType.YML
    audio_segments: List[AudioChainArtifactFile] | None = None
    transcript: str | None = None
    audio_mosiac: AudioChainArtifactFile | None = None
    noise_blended_audio: AudioChainArtifactFile | None = None

    def __init__(self, **data):
        super().__init__(**data)

    def save_file(self):
        if self.audio_segments:
            for seg in self.audio_segments:
                if seg is not None:
                    logging.info(
                        f"Saving audio segment to: {seg.base_path}"
                    )  # ✅ Add this

                    seg.save_file()

        if self.audio_mosiac is not None:
            logging.info(
                f"Saving mosaic to: {self.audio_mosiac.base_path}"
            )  # ✅ Add this

            self.audio_mosiac.save_file()

        if self.noise_blended_audio is not None:
            logging.info(
                f"Saving blended to: {self.noise_blended_audio.base_path}"
            )  # ✅ Add this
            self.noise_blended_audio.save_file()

        if self.base_path is None:
            self.base_path = os.path.join(
                os.getenv("AGENT_WORK_PRODUCT_BASE_PATH", "artifacts"), self.thread_id
            )
        file_path_obj = Path(self.base_path) / "yml"
        file_path_obj.mkdir(parents=True, exist_ok=True)
        file_obj = file_path_obj / self.file_name

        data_to_save = {
            "transcript": self.transcript,
            "audio_segments": (
                [seg.file_path for seg in self.audio_segments]
                if self.audio_segments
                else []
            ),
            "audio_mosiac": self.audio_mosiac.file_path if self.audio_mosiac else None,
            "noise_blended_audio": (
                self.noise_blended_audio.file_path if self.noise_blended_audio else None
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
            "thread_id": self.thread_id,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "chain_artifact_type": ChainArtifactType.EVP_ARTIFACT.value,
            "chain_artifact_file_type": "yaml",
            "transcript": self.transcript,
            "audio_segments": (
                [seg.file_path for seg in self.audio_segments]
                if self.audio_segments
                else []
            ),
            "audio_mosiac": self.audio_mosiac.file_path if self.audio_mosiac else None,
            "noise_blended_audio": (
                self.noise_blended_audio.file_path if self.noise_blended_audio else None
            ),
        }


if __name__ == "__main__":
    with open("/Volumes/LucidNonsense/White/tests/mocks/mock.wav", "rb") as f:
        audio_bytes = f.read()
    with open(
        f"{os.getenv('AGENT_MOCK_DATA_PATH')}/black_evp_artifact_mock.yml", "r"
    ) as f:
        data = yaml.safe_load(f)
        evp_artifact = EVPArtifact(**data)
        evp_artifact.audio_segments = [
            AudioChainArtifactFile(
                thread_id="test_thread_id",
                chain_artifact_type="unknown",
                chain_artifact_file_type="wav",
                base_path="/Volumes/LucidNonsense/White/chain_artifacts/",
                artifact_name="test_audio_artifact",
                sample_rate=44100,
                duration=5.0,
                audio_bytes=audio_bytes,
                channels=2,
            )
        ]
        evp_artifact.audio_mosiac = AudioChainArtifactFile(
            thread_id="test_thread_id",
            chain_artifact_type="unknown",
            chain_artifact_file_type="wav",
            base_path="/Volumes/LucidNonsense/White/chain_artifacts/",
            artifact_name="test_audio_artifact",
            sample_rate=44100,
            duration=5.0,
            audio_bytes=audio_bytes,
            channels=2,
        )
        evp_artifact.noise_blended_audio = AudioChainArtifactFile(
            thread_id="test_thread_id",
            chain_artifact_type="unknown",
            chain_artifact_file_type="wav",
            base_path="/Volumes/LucidNonsense/White/chain_artifacts/",
            artifact_name="test_audio_artifact",
            sample_rate=44100,
            duration=5.0,
            audio_bytes=audio_bytes,
            channels=2,
        )
        evp_artifact.save_file()
        print(evp_artifact.flatten())
