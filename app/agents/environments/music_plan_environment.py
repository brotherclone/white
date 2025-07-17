import gymnasium as gym
import numpy as np
import yaml
import os

from gymnasium import spaces
from stable_baselines3 import PPO
from typing import Optional, Tuple, Any


class MusicPlanEnvironment(gym.Env):

    def __init__(self, reference_plans_dir: str):
        super().__init__()

    def _load_reference_plans(self, reference_plans_dir: str) -> dict:
        pass

    def reset(self):
        pass
    def _color_to_state(self):
        pass

    def step(self):
        pass

    def _action_to_plan(self):
        pass

    def _calculate_reward(self):
        pass

    def _similarity_to_reference(self):
        pass

    def _update_state(self):
        pass