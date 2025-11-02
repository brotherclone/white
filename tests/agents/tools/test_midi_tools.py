from app.agents.tools.midi_tools import MidiProcessor


class FakeState:
    def __init__(self, cut_up_fragments, session_id, midi_data_attr=False):
        self.cut_up_fragments = cut_up_fragments
        self.session_id = session_id
        if midi_data_attr:
            self.midi_data = {"placeholder": True}


def test_midi_processor_adds_midi_data_when_missing():
    state = FakeState(cut_up_fragments=["frag1", "frag2", "frag3"], session_id="sess123")
    proc = MidiProcessor()
    out = proc(state)

    assert hasattr(out, "midi_data")
    md = out.midi_data
    assert md["tracks"] == 3
    assert md["fragments_mapped"] == 3
    assert md["estimated_duration"] == 3 * 0.5
    assert md["file_path"] == "generated_music_sess123.mid"
    assert md["bpm"] == 120
    assert md["time_signature"] == "4/4"
    assert md["key"] == "C minor"


def test_midi_processor_overwrites_existing_midi_data():
    state = FakeState(cut_up_fragments=["a"], session_id="X", midi_data_attr=True)
    assert state.midi_data == {"placeholder": True}
    proc = MidiProcessor()
    out = proc(state)

    # Ensure the placeholder was replaced by new structured midi_data
    assert isinstance(out.midi_data, dict)
    assert out.midi_data.get("placeholder") is None
    assert out.midi_data["tracks"] == 1
    assert out.midi_data["file_path"] == "generated_music_X.mid"


def test_midi_processor_handles_zero_fragments():
    state = FakeState(cut_up_fragments=[], session_id="zero")
    proc = MidiProcessor()
    out = proc(state)

    md = out.midi_data
    assert md["tracks"] == 0
    assert md["fragments_mapped"] == 0
    assert md["estimated_duration"] == 0.0
    assert md["file_path"] == "generated_music_zero.mid"