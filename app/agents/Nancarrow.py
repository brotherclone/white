from transformers import MusicgenForConditionalGeneration, AutoProcessor, AutoFeatureExtractor
from typing import Any

from app.agents.BaseRainbowAgent import BaseRainbowAgent


class Nancarrow(BaseRainbowAgent):

    model: Any = None
    extractor: Any = None

    def __init__(self, **data):
        super().__init__(**data)
        self.model = None
        self.extractor = None
        self.llm_model_name = "facebook/musicgen-small"

    def initialize(self):
        self.model = MusicgenForConditionalGeneration.from_pretrained(self.llm_model_name)
        self.extractor =AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        self.agent_state = None

    def generate_midi_from_plan(self, plan):
       pass

    def visualize_midi(self, midi_data):
        pass

    def play_preview(self, midi_data):
        pass