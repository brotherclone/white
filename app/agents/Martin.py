from transformers import (
    AutoModel,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    pipeline
)
from typing import Optional, Any

from app.agents.BaseRainbowAgent import BaseRainbowAgent


class Martin(BaseRainbowAgent):
    processor: Optional[Any] = None
    generator: Optional[Any] = None
    analyzer: Optional[Any] = None

    def __init__(self, **data):
        super().__init__(**data)
        if not self.analyzer_name:
            self.analyzer_name = "facebook/wav2vec2-base-960h"
        if not self.processor_name:
            self.processor_name = "facebook/wav2vec2-base-960h"

        self.processor = None
        self.generator = None
        self.analyzer = None

    def initialize(self):
        try:
            # Use more specific loading methods for audio models
            self.processor = AutoFeatureExtractor.from_pretrained(self.processor_name)
            self.analyzer = AutoModelForAudioClassification.from_pretrained(
                self.analyzer_name,
                num_labels=10  # Adjust based on your classification needs
            ).to(self.device)
            print("Audio models loaded successfully")
        except Exception as e:
            print(f"Error loading audio models: {e}")