import random

from app.agents.BaseRainbowAgent import BaseRainbowAgent
from app.enums.rainbow_color import RainbowColor


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

    def train_on_plans(self, **kwargs):
        pass

    def train_on_experience(self, **kwargs):
        pass

    def get_random_best_plan(self):
        pass

    def append_implemented_plan(self, plan):
        pass