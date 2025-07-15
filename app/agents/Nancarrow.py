from transformers import AutoModel

from app.agents.BaseRainbowAgent import BaseRainbowAgent


class Nancarrow(BaseRainbowAgent):

    def __init__(self, **data):
        super().__init__(data)
        self.model = None

    def initialize(self):
        self.model = AutoModel.from_pretrained(self.llm_model_name).to(self.device)
        self.agent_state = None