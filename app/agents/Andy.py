from transformers import (
    MusicgenForConditionalGeneration,
    AutoProcessor,
    AutoFeatureExtractor
)
from app.agents.BaseRainbowAgent import BaseRainbowAgent
from app.agents.Dorthy import Dorthy
from app.agents.Martin import Martin
from app.agents.Nancarrow import Nancarrow
from app.agents.Subutai import Subutai
from typing import Optional

class Andy(BaseRainbowAgent):
    lyrics_agent: Optional[Dorthy] = None
    audio_agent: Optional[Martin] = None
    midi_agent: Optional[Nancarrow] = None
    planning_agent: Optional[Subutai] = None

    def __init__(self, **data):
        super().__init__(**data)
        self.lyrics_agent = Dorthy(
            tokenizer_name="gpt2",
            llm_model_name="gpt2",
            generator_name="gpt2"
        )
        self.audio_agent = Martin(
            analyzer_name="facebook/wav2vec2-base-960h",
            processor_name="facebook/wav2vec2-base-960h"
        )
        self.midi_agent = Nancarrow(
            llm_model_name="bert-base-uncased",
        )
        self.planning_agent = Subutai()

    def initialize(self):
        training_path = "/Volumes/LucidNonsense/White/training"
        self.lyrics_agent.load_training_data(training_path)
        self.lyrics_agent.initialize()
        self.audio_agent.load_training_data(training_path)
        self.audio_agent.initialize()
        self.midi_agent.load_training_data(training_path)
        self.midi_agent.initialize()
        self.planning_agent.initialize()