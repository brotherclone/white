from pydantic import BaseModel
from app.agents.states.main_agent_state import MainAgentState
from typing import Dict, Optional


class MidiProcessor(BaseModel):

    def __init__(self, **data):
        super().__init__(**data)

    def __call__(self, state: MainAgentState) -> MainAgentState:
        print("🎵 MIDI GENERATOR: Converting fragments to MIDI...")

        # Create mock MIDI data based on cut-up fragments
        midi_data = {
            "tracks": len(state.cut_up_fragments),
            "fragments_mapped": len(state.cut_up_fragments),
            "estimated_duration": len(state.cut_up_fragments) * 0.5,  # seconds
            "mapping_strategy": "Fragment length → note duration, syllable count → pitch",
            "file_path": f"generated_music_{state.session_id}.mid",
            "bpm": 120,
            "time_signature": "4/4",
            "key": "C minor"
        }

        # Add MIDI data to state - need to add this field to MainAgentState
        if not hasattr(state, 'midi_data'):
            state.midi_data = midi_data
        else:
            state.midi_data = midi_data

        print(f"Generated MIDI mapping for {midi_data['tracks']} tracks")
        return state
