from transformers import AutoModel, pipeline
from typing import Any
from app.agents.BaseRainbowAgent import BaseRainbowAgent
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification


class Martin(BaseRainbowAgent):
    processor: Any = None
    generator: Any = None
    analyzer: Any = None
    model: Any = None

    def __init__(self, **data):
        super().__init__(**data)
        self.processor = None
        self.generator = None
        self.analyzer = None

    def initialize(self):
        self.agent_state = None
        if not self.llm_model_name:
            self.llm_model_name = "gpt2"
        self.model = AutoModel.from_pretrained(
            self.llm_model_name,
            use_safetensors=True
        ).to(self.device)
        if not self.generator_name:
            self.generator_name = "bert-base-uncased"
        self.generator = AutoModel.from_pretrained(
            self.generator_name,
            use_safetensors=True
        ).to(self.device)

        # Use feature extractor instead of processor
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-ks")
        self.analyzer = Wav2Vec2ForSequenceClassification.from_pretrained(
            "superb/wav2vec2-base-superb-ks",
            use_safetensors=True
        ).to(self.device)