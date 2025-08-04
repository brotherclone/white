from transformers import AutoModel, pipeline
from typing import Any
from app.agents.BaseRainbowAgent import BaseRainbowAgent


class Martin(BaseRainbowAgent):

    processor: Any = None
    generator: Any = None
    analyzer: Any = None

    def __init__(self, **data):
        super().__init__(**data)
        self.processor = None
        self.generator = None
        self.analyzer = None

    def initialize(self):
        self.analyzer = pipeline("audio-classification", model=self.analyzer_name)
        if not self.generator_name:
            self.generator_name = "bert-base-uncased"  # or your desired model
        self.generator = AutoModel.from_pretrained(self.generator_name).to(self.device)
        self.processor = AutoModel.from_pretrained(self.processor_name)