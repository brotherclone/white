from typing import Any

from transformers import AutoTokenizer, AutoModel, pipeline
from app.agents.BaseRainbowAgent import BaseRainbowAgent


class Dorthy(BaseRainbowAgent):
    generator: Any = None
    model: Any = None
    tokenizer: Any = None
    lyrics_data: Any = None
    vector_store: Any = None

    def __init__(self, **data):
        super().__init__(**data)
        self.generator = None
        self.model = None
        self.tokenizer = None
        self.embeddings = None
        self.vector_store = None
        self.training_data = None
        self.llm_model_name = "gpt2"
        self.tokenizer_name = "gpt2"
        self.generator_name = "gpt2"

    def initialize(self):
        self.agent_state = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModel.from_pretrained(
            self.llm_model_name,
            use_safetensors=True
        )
        self.generator = pipeline("text-generation", model=self.generator_name)

    def _process_agent_specific_data(self) -> None:
        if self.training_data is None:
            print("No training data available to process.")
            return
        print(self.training_data.columns)

    def process(self):
        # Processing logic specific to Dorthy
        pass
