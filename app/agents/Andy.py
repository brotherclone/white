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
        self.agent_state = None
        training_path = "/Volumes/LucidNonsense/White/training"
        self.lyrics_agent.load_training_data(training_path)
        self.lyrics_agent.initialize()
        self.audio_agent.load_training_data(training_path)
        self.audio_agent.initialize()
        self.midi_agent.load_training_data(training_path)
        self.midi_agent.initialize()
        self.plan_agent.load_training_data(training_path)
        self.plan_agent.initialize()

    def create_rainbow_plan(self, resource_id=None):
        pass

    def generate_song_with_chords(self, plan_or_color=None):
        """Generate a complete song including chord progressions"""

        # Step 1: Generate basic plan
        if plan_or_color is None:
            plan = self.plan_agent.generate_plans()
        else:
            plan = self.plan_agent.generate_plans(rainbow_color=plan_or_color)

        # Step 2: Convert plan to song structure
        song_plan = self.plan_agent.env.plan_to_rainbow_song_plan(plan)

        # Step 3: Generate chord progressions for each section
        chord_progressions = []
        previous_progression = None

        for section in song_plan.structure:
            progression = self.chord_generator.generate_progression_for_section(
                section=section,
                key=song_plan.key,
                mood_tags=song_plan.moods or [],
                previous_progression=previous_progression
            )
            chord_progressions.append(progression)
            previous_progression = progression

        return {
            "basic_plan": plan,
            "song_plan": song_plan,
            "chord_progressions": chord_progressions,
            "chord_charts": [prog.to_chord_chart() for prog in chord_progressions]
        }
