import os
import random
from abc import ABC
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, ConfigDict

from app.agents.tools.infranym_audio_encoder import InfranymAudioEncoder
from app.structures.artifacts.audio_artifact_file import AudioChainArtifactFile
from app.structures.artifacts.infranym_voice_composition import InfranymVoiceComposition
from app.structures.artifacts.infranym_voice_layer import InfranymVoiceLayer
from app.structures.enums.infranym_voice_profile import InfranymVoiceProfile

load_dotenv()

# ToDo: Finish implementation


class InfranymAudioArtifact(AudioChainArtifactFile, ABC):
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
        self.encoder = InfranymAudioEncoder(
            output_dir=f"{os.getenv('AGENT_WORK_PRODUCT_BASE_PATH')}/infranym_audio_encoder"
        )
        self.surface_layer = self._get_random_synth_voice(
            is_reversed=False, is_filtered=False
        )
        self.reverse_layer = self._get_random_synth_voice(
            is_reversed=True, is_filtered=False
        )
        self.submerged_layer = self._get_random_synth_voice(
            is_reversed=False, is_filtered=True
        )

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

    def flatten(self):
        pass

    def for_prompt(self):
        pass

    def save_file(self):
        pass


if __name__ == "__main__":
    pass
