import random
import os

from stable_baselines3 import PPO
from app.agents.BaseRainbowAgent import BaseRainbowAgent
from app.objects.rainbow_color import RainbowColor
from app.agents.environments.music_plan_environment import MusicPlanEnvironment
from typing import Any

class Subutai(BaseRainbowAgent):

    env: Any = None
    model: Any = None
    is_trained: bool = False

    def __init__(self, **data):
        super().__init__(**data)
        self.env = None
        self.model = None
        self.is_trained = False

    def initialize(self, reference_plans_dir: str = None, **kwargs) -> None:
        # First initialize the environment
        self.env = MusicPlanEnvironment()
        # Then create the model using the environment
        self.model = PPO("MlpPolicy", self.env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64,
                         n_epochs=10, gamma=0.99, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2)

    def train(self, total_timesteps: int = 20000, save_path=None, **kwargs) -> None:
        if self.model is None:
            raise ValueError("Model is not initialized. Please call initialize() before training.")
        self.model.learn(total_timesteps=total_timesteps)
        self.is_trained = True
        if save_path is not None:
            self.model.save(save_path)

    def load(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
        self.model = PPO.load(model_path, env=self.env)
        self.is_trained = True

    def _process_agent_specific_data(self) -> None:
        pass

    @staticmethod
    def get_random_rainbow_color() -> RainbowColor:
        return random.choice(list(RainbowColor))

    def generate_plans(self, **kwargs):
        plan_rainbow_color = kwargs.get('rainbow_color')
        if plan_rainbow_color is None:
            plan_rainbow_color = self.get_random_rainbow_color()

        if not self.is_trained or self.model is None:
            # Fallback to random plans if not trained
            return self._generate_random_plan(plan_rainbow_color)

        # Use the model to generate a plan
        obs, _ = self.env.reset(options={"color": plan_rainbow_color})
        action, _ = self.model.predict(obs, deterministic=False)

        # Convert action to plan
        plan = self.env._action_to_plan(action)
        plan["color"] = str(plan_rainbow_color)

        return plan

    def _generate_random_plan(self, rainbow_color):
        """Generate a random plan when model isn't trained"""
        action = self.env.action_space.sample()
        plan = self.env._action_to_plan(action)
        plan["color"] = str(rainbow_color)
        return plan