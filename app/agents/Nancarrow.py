from transformers import AutoModel
from typing import Any
from app.agents.BaseRainbowAgent import BaseRainbowAgent
from app.objects.song_plan import RainbowSongPlan


class Nancarrow(BaseRainbowAgent):

    model: Any = None
    midi_data: Any = None

    def __init__(self, **data):
        super().__init__(**data)
        self.model = None

    def initialize(self):
        self.model = AutoModel.from_pretrained(self.llm_model_name).to(self.device)
        self.agent_state = None