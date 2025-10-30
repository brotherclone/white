from typing import List

from app.agents.models.text_chain_artifact_file import TextChainArtifactFile
from app.agents.models.audio_chain_artifact_file import AudioChainArtifactFile
from app.agents.models.base_chain_artifact import ChainArtifact


class EVPArtifact(ChainArtifact):

    audio_segments: List[AudioChainArtifactFile] | None = None
    transcript: TextChainArtifactFile | None = None
    audio_mosiac: AudioChainArtifactFile | None = None
    noise_blended_audio: AudioChainArtifactFile | None = None
    thread_id: str

    def __init__(self, **data):
        super().__init__(**data)

    #ToDo implement to clean up segments and blended - keep mosaic and transcript
    def clean_temp_files(self):
        pass