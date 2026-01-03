import os
import random
import yaml

from abc import ABC
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field, ConfigDict

from app.agents.tools.infranym_audio_encoder import InfranymAudioEncoder
from app.structures.artifacts.audio_artifact_file import AudioChainArtifactFile
from app.structures.artifacts.infranym_voice_composition import InfranymVoiceComposition
from app.structures.artifacts.infranym_voice_layer import InfranymVoiceLayer
from app.structures.enums.chain_artifact_type import ChainArtifactType
from app.structures.enums.infranym_voice_profile import InfranymVoiceProfile

load_dotenv()


class InfranymAudioArtifact(AudioChainArtifactFile, ABC):
    chain_artifact_type: ChainArtifactType = ChainArtifactType.INFRANYM_AUDIO
    title: Optional[str] = Field(
        default="Infranym Audio", description="Title of the audio"
    )
    secret_word: str = Field(
        default="",
        description="Secret word",
    )
    bpm: int = Field(default=100, description="BPM of the audio")
    key: Optional[str] = Field(default=None, description="Key signature of the audio")
    encoder: InfranymAudioEncoder | None = None
    composition: Optional[InfranymVoiceComposition] = Field(
        default=None, description="Composition to be synthesized into audio."
    )
    surface_layer: Optional[InfranymVoiceLayer] = Field(
        default=None, description="Surface layer of the composition."
    )
    reverse_layer: Optional[InfranymVoiceLayer] = Field(
        default=None, description="Reverse layer of the composition."
    )
    submerged_layer: Optional[InfranymVoiceLayer] = Field(
        default=None, description="Submerged layer of the composition."
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        artifact_dir = self.get_artifact_path(with_file_name=False, create_dirs=True)
        self.encoder = InfranymAudioEncoder(output_dir=str(artifact_dir))
        self.surface_layer = self._get_random_synth_voice(
            is_reversed=False, is_filtered=False
        )
        self.reverse_layer = self._get_random_synth_voice(
            is_reversed=True, is_filtered=False
        )
        self.submerged_layer = self._get_random_synth_voice(
            is_reversed=False, is_filtered=True
        )
        self.composition = self._generate_composition()

    def _get_random_synth_voice(
        self, is_reversed: bool, is_filtered
    ) -> InfranymVoiceLayer:
        a_rate = random.randint(50, 150)
        a_pitch = random.uniform(0.0, 0.99)
        a_volume = random.uniform(0.0, -20.00)
        a_top_frq = random.uniform(200.0, 1000.0)
        a_bottom_frq = random.uniform(20.0, 201.0)
        a_voice_profile = random.choice(list(InfranymVoiceProfile))
        a_pan = random.uniform(-1.0, 1.0)
        if is_filtered:
            profile = InfranymVoiceLayer(
                text=self.secret_word,
                voice_profile=a_voice_profile,
                rate=a_rate,
                pitch=a_pitch,
                volume_db=a_volume,
                freq_filter=(a_top_frq, a_bottom_frq),
                reverse=is_reversed,
                stereo_pan=a_pan,
            )
        else:
            profile = InfranymVoiceLayer(
                text=self.secret_word,
                voice_profile=a_voice_profile,
                rate=a_rate,
                pitch=a_pitch,
                volume_db=a_volume,
                reverse=is_reversed,
                stereo_pan=a_pan,
            )
        return profile

    def _generate_composition(self) -> InfranymVoiceComposition:
        composition = InfranymVoiceComposition(
            title=f"{self.thread_id}_audio_infranym",
            tempo_bpm=self.bpm,
            key_signature=self.key,
            surface_layer=self.surface_layer,
            reverse_layer=self.reverse_layer,
            submerged_layer=self.submerged_layer,
            metadata={
                "puzzle_solution": self.secret_word,
                "color_agent": "Indigo",
                "album": "Untitled White Album",
            },
        )
        return composition

    def flatten(self):
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}
        return {
            **parent_data,
            "secret_word": self.secret_word,
            "bpm": self.bpm,
            "key": self.key,
            "composition": self.composition.model_dump(),
            "surface_layer": self.surface_layer.model_dump(),
            "reverse_layer": self.reverse_layer.model_dump(),
            "submerged_layer": self.submerged_layer.model_dump(),
            "title": self.title,
            "metadata": self.composition.metadata,
        }

    def for_prompt(self):
        prompt_parts = []
        if self.title:
            prompt_parts.append(f"Audio: {self.title}")
        return "\n".join(prompt_parts)

    def save_file(self):
        filename_stem = self.artifact_name
        self.encoder.encode_composition(
            self.composition, output_filename=filename_stem, export_layers=True
        )


if __name__ == "__main__":
    with open("/Volumes/LucidNonsense/White/tests/mocks/mock.wav", "rb") as f:
        audio_bytes = f.read()
    with open(
        f"{os.getenv('AGENT_MOCK_DATA_PATH')}/indigo_infranym_audio_mock.yml", "r"
    ) as f:
        data_i = yaml.safe_load(f)
        data_i["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
        data_i["audio_bytes"] = audio_bytes
        infranym_audio_artifact = InfranymAudioArtifact(**data_i)
        infranym_audio_artifact.save_file()
        print(infranym_audio_artifact.flatten())
        p = infranym_audio_artifact.for_prompt()
        print(p)
