from typing import Optional
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

    def generate_from_your_catalog(self, rainbow_color=None, target_moods=None):
        """Generate new music based on patterns from your 10 years of work"""

        print("🎵 Generating music from your catalog...")
        print("=" * 50)

        # Step 1: Generate plan using reinforcement learning
        plan = self.planning_agent.generate_plans(rainbow_color=rainbow_color)
        song_plan = self.planning_agent.env.plan_to_rainbow_song_plan(plan)

        print(f"🎯 Plan: {song_plan.key} at {song_plan.bpm} BPM")
        print(f"🌈 Color: {song_plan.rainbow_color}")
        print(f"🎭 Moods: {song_plan.moods}")

        # Step 2: Generate content for each section using your training data
        results = []

        for section in song_plan.structure:
            print(f"\n🎼 Processing: {section.section_name}")

            section_result = {
                'section_name': section.section_name,
                'section_description': section.section_description
            }

            # Generate lyrics if it's a vocal section
            vocal_sections = ['verse', 'chorus', 'bridge']
            if any(vs in section.section_name.lower() for vs in vocal_sections):
                lyrics_result = self.lyrics_agent.generate_lyrics_from_training_data(
                    target_mood=song_plan.moods or target_moods or ['mysterious'],
                    rainbow_color=str(song_plan.rainbow_color),
                    section_name=section.section_name,
                    key=song_plan.key
                )
                section_result['lyrics'] = lyrics_result
                print(f"  🎤 Lyrics from: {lyrics_result['inspiration_source']}")

            # Generate MIDI patterns
            midi_result = self.midi_agent.generate_midi_from_training_data(
                target_mood=song_plan.moods or target_moods or ['mysterious'],
                rainbow_color=str(song_plan.rainbow_color),
                section_name=section.section_name,
                key=song_plan.key,
                bpm=song_plan.bpm
            )
            section_result['midi'] = midi_result
            print(f"  🎹 MIDI from: {midi_result['inspiration_source']}")

            results.append(section_result)

        return {
            'plan': plan,
            'song_plan': song_plan,
            'sections': results,
            'training_data_used': True
        }
