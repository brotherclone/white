import random
import numpy as np
import os

from stable_baselines3 import PPO

from app.agents.BaseRainbowAgent import BaseRainbowAgent
from app.objects.rainbow_color import RainbowColor

class Subutai(BaseRainbowAgent):

    def __init__(self, **data):
        super().__init__(**data)

    def initialize(self):
        pass

    def _process_agent_specific_data(self) -> None:
        pass

    @staticmethod
    def get_random_rainbow_color()-> RainbowColor:
        return random.choice(list(RainbowColor))

    def generate_plans(self, **kwargs):
        if kwargs.rainbow_color is None:
           col = self.get_random_rainbow_color()
        else:
            col = kwargs.rainbow_color
        pass

    def _generate_random_plan(self):
        pass


