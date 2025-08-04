
from typing import Any
from app.agents.BaseRainbowAgent import BaseRainbowAgent


class Subutai(BaseRainbowAgent):

    generator: Any = None
    model: Any = None
    tokenizer: Any = None
    plan_data: Any = None
    vector_store: Any = None


    def __init__(self, **data):
        super().__init__(**data)
        self.generator = None
        self.model = None
        self.tokenizer = None
        self.embeddings = None
        self.vector_store = None
        self.training_data = None

    def initialize(self):
        pass

    def _process_agent_specific_data(self) -> None:
        pass

    def process(self):
        pass

    def generate_plan(self):
        pass

    def critique_plan(self):
        pass
