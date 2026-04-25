from abc import ABC

from white_core.artifacts.base_artifact import ChainArtifact


class MidiArtifactFile(ChainArtifact, ABC):

    def __init__(self, **data):
        super().__init__(
            **data,
        )
