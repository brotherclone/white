import gymnasium as gym
import numpy as np
import yaml
import os

from gymnasium import spaces
from stable_baselines3 import PPO
from typing import Optional, Tuple, Any

REFERENCE_PLANS_DIR = "/Volumes/LucidNonsense/White/plans/reference"


class MusicPlanEnvironment(gym.Env):

    def __init__(self):
        super().__init__()
        self.reference_plans = self._load_reference_plans(REFERENCE_PLANS_DIR)
        self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.float32)
        self.current_state = None
        self.current_color = None

    @staticmethod
    def _load_reference_plans(reference_plans_dir: str) -> dict or None:
        references = dict(positive=[], negative=[])
        reviewed_dir = os.path.join(reference_plans_dir, "reviewed")
        absolute_reviewed_dir_path = os.path.abspath(reviewed_dir)
        if os.path.exists(absolute_reviewed_dir_path):
            for filename in os.listdir(absolute_reviewed_dir_path):
                if filename.endswith(".yml"):
                    yml_path = os.path.join(absolute_reviewed_dir_path, filename)
                    with open(yml_path, "r") as f:
                        a_plan = yaml.safe_load(f)
                        ratings = []
                        for key in a_plan:
                            if key.endswith("_feedback") and a_plan[key] and a_plan[key].get("rating"):
                                ratings.append(a_plan[key]["rating"])
                        if ratings:
                            avg_rating = sum(ratings) / len(ratings)
                            if avg_rating >= 7.0:
                                references["positive"].append(a_plan)
                            else:
                                references["negative"].append(a_plan)
            return references
        else:
            print(f"Reference plans directory '{absolute_reviewed_dir_path}' does not exist.")
            return None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Extract color from options if provided
        self.current_color = None
        if options and "color" in options:
            self.current_color = options["color"]

        # Initialize state based on color or randomly
        if self.current_color:
            self.current_state = self._color_to_state(self.current_color)
        else:
            self.current_state = self.observation_space.sample()

        return self.current_state, {}

    def _color_to_state(self, color) -> np.ndarray:
        """Convert rainbow color to state representation"""
        color_values = {
            "Z": -1.0,  # Black
            "R": -0.75,  # Red
            "O": -0.5,  # Orange
            "Y": -0.25,  # Yellow
            "G": 0.0,  # Green
            "B": 0.25,  # Blue
            "I": 0.5,  # Indigo
            "V": 0.75,  # Violet
            "A": 1.0  # White
        }

        base_value = color_values.get(str(color), 0)
        state = np.ones(self.observation_space.shape) * base_value
        state += np.random.normal(0, 0.05, self.observation_space.shape)
        return np.clip(state, -1, 1)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action and return new state, reward, etc."""
        # Convert action to a plan
        generated_plan = self._action_to_plan(action)

        # Calculate reward based on similarity to reference plans
        reward = self._calculate_reward(generated_plan)

        # Update state
        self.current_state = self._update_state(action)

        # Always False since this is an ongoing creative process
        terminated = False
        truncated = False

        info = {"plan": generated_plan}

        return self.current_state, reward, terminated, truncated, info

    @staticmethod
    def _action_to_plan (action: np.ndarray) -> dict:
        """Convert action vector to a music plan"""
        # Get note (first 12 possibilities)
        note_index = int(action[1] * 12)
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        note = notes[note_index]

        # Add mode (using another part of the action value)
        # Use a smaller portion of the action range for mode selection
        mode_value = (action[1] * 12) % 1  # Get fractional part
        if mode_value < 0.7:  # 70% chance of common modes
            mode = "major" if mode_value < 0.35 else "minor"
        else:  # 30% chance of less common modes
            rare_modes = ["dorian", "phrygian", "lydian", "mixolydian", "locrian"]
            rare_index = int(mode_value * 10) % len(rare_modes)
            mode = rare_modes[rare_index]

        plan = {
            "tempo": int(action[0] * 100 + 60),  # 60-160 BPM
            "key": f"{note} {mode}",  # Complete musical key with mode
            "sections": int(action[2] * 5) + 2,  # 2-7 sections
            "motifs": [float(x) for x in action[3:6]],  # Motif parameters
            "instruments": [int(x * 10) for x in action[6:]]  # Instrument selection parameters
        }
        return plan

    def _calculate_reward(self, plan: dict) -> float:
        """Calculate reward based on similarity to positive examples and difference from negative ones"""
        if not self.reference_plans:
            return 0.0

        positive_similarity = self._similarity_to_reference(plan, self.reference_plans["positive"])
        negative_similarity = self._similarity_to_reference(plan, self.reference_plans["negative"])

        # Reward similarity to positive examples and penalize similarity to negative ones
        return positive_similarity - (1.5 * negative_similarity)

    @staticmethod
    def _similarity_to_reference(plan: dict, references: list) -> float:
        """Calculate similarity between a plan and reference plans"""
        if not references:
            return 0.0

        similarity_sum = 0
        for ref_plan in references:
            # Calculate similarity
            matches = 0
            total = 0
            for key, value in plan.items():
                if key in ref_plan:
                    total += 1
                    if isinstance(value, (list, np.ndarray)):
                        # For lists/arrays, calculate overlap
                        ref_val = ref_plan[key]
                        if isinstance(ref_val, (list, np.ndarray)) and len(value) == len(ref_val):
                            similarity = 1 - min(1, np.mean(np.abs(np.array(value) - np.array(ref_val))))
                            matches += similarity
                    else:
                        # For scalar values, check exact match
                        if value == ref_plan[key]:
                            matches += 1

            similarity_sum += matches / max(1, total)

        return similarity_sum / len(references)

    def _update_state(self, action: np.ndarray) -> np.ndarray:
        """Update state based on action"""
        # Blend the current state with action influence
        new_state = 0.8 * self.current_state + 0.2 * (action * 2 - 1)
        return np.clip(new_state, -1, 1)

    def plan_to_rainbow_song_plan(self, generated_plan: dict) -> 'RainbowSongPlan':
        """Convert simplified RL plan to full RainbowSongPlan object"""
        from app.objects.song_plan import RainbowSongPlan
        from app.objects.rainbow_color import RainbowColor
        from app.objects.rainbow_song_meta import RainbowSongStructureModel

        plan = RainbowSongPlan()

        # Handle key with mode (already properly formatted)
        if "key" in generated_plan:
            plan.key = generated_plan["key"]

        if "tempo" in generated_plan:
            plan.bpm = generated_plan["tempo"]

        # Convert motif parameters to structure
        if "motifs" in generated_plan and "sections" in generated_plan:
            complexity, variation, repetition = generated_plan["motifs"]
            section_count = generated_plan["sections"]

            # Create structure based on motif parameters
            plan.structure = []
            for i in range(section_count):
                # Determine section type based on position and parameters
                if i == 0:
                    section_type = "Intro"
                elif i == section_count - 1:
                    section_type = "Outro"
                elif i % 3 == 1 and complexity > 0.5:
                    section_type = "Chorus"
                elif i % 4 == 0 and repetition > 0.7:
                    section_type = "Bridge"
                else:
                    section_type = "Verse"

                # Create section
                section = RainbowSongStructureModel(
                    section_name=f"{section_type} {i//3 + 1}" if section_type in ["Verse", "Chorus"] else section_type,
                    sequence=i+1,
                    section_description=f"Generated with complexity={complexity:.2f}, variation={variation:.2f}"
                )
                plan.structure.append(section)

        return plan
