import random
import os

from typing import Any, Optional
from app.agents.BaseRainbowAgent import BaseRainbowAgent
from app.agents.Dorthy import Dorthy
from app.agents.Martin import Martin
from app.agents.Nancarrow import Nancarrow
from app.agents.Subutai import Subutai

class Andy(BaseRainbowAgent):

    lyrics_agent: Optional[Dorthy] = None
    audio_agent: Optional[Martin] = None
    midi_agent: Optional[Nancarrow] = None
    plan_agent: Optional[Subutai] = None


    def __init__(self, **data):
        super().__init__(**data)
        self.lyrics_agent = Dorthy()
        self.audio_agent = Martin()
        self.midi_agent = Nancarrow()
        self.plan_agent = Subutai()


    def initialize(self):
        training_path = "/Volumes/LucidNonsense/White/training"
        # self.lyrics_agent.load_training(training_path)
        # self.lyrics_agent.initialize()
        # self.audio_agent.load_training(training_path)
        # self.audio_agent.initialize()
        # self.midi_agent.load_training(training_path)
        # self.midi_agent.initialize()
        # self.plan_agent.load_training(training_path)
        # self.plan_agent.initialize()



    def create_rainbow_plan(self, resource_id=None):
        pass