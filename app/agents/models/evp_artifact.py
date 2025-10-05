from typing import List

from app.agents.models.text_chain_artifact_file import TextChainArtifactFile
from app.agents.models.audio_chain_artifact_file import AudioChainArtifactFile
from app.agents.models.base_chain_artifact import ChainArtifact


class EVPArtifact(ChainArtifact):

    audio_segments: List[AudioChainArtifactFile]
    transcript: TextChainArtifactFile
    audio_mosiac: AudioChainArtifactFile
    noise_blended_audio: AudioChainArtifactFile
    thread_id: str

    def __init__(self, **data):
        super().__init__(**data)