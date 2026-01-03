from abc import ABC

from app.structures.artifacts.base_artifact import ChainArtifact


# ToDo: Finish implementation
class InfranymTextArtifact(ChainArtifact, ABC):

    def __init__(self, **data):
        super().__init__(**data)

    def flatten(self):
        pass

    def for_prompt(self):
        pass

    def save_file(self):
        pass
