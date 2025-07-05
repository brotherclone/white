from transformers import AutoModel, pipeline

from app.agents.BaseRainbowAgent import BaseRainbowAgent


class Martin(BaseRainbowAgent):

    def __init__(self, **data):
        super().__init__(data)
        self.processor = None
        self.generator = None
        self.analyzer = None

    def initialize(self):
        self.analyzer = pipeline("audio-classification", model=self.analyzer_name)
        self.generator = AutoModel.from_pretrained(self.generator_name).to(self.device)
        self.processor = AutoModel.from_pretrained(self.processor_name)