import torch
import os
import json

from app.agents.Andy import Andy
from app.objects.rainbow_color import RainbowColor


TRAINING_PATH = "/Volumes/LucidNonsense/White/training"
MODELS_PATH = "./models"  # Directory to save/load models
PLAN_OUTPUT_PATH = "./plans/unreviewed"  # Directory to save generated plans

def try_andy():
    print("\n=== Testing Andy Coordinator Agent ===")
    andy = Andy()
    andy.initialize()
    plan = andy.planning_agent.generate_plans(rainbow_color=RainbowColor.Z)
    print("Generated plan:")
    for key, value in plan.items():
        print(f"  {key}: {value}")




if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to: {device}")
    try_andy()