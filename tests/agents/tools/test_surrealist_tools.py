from app.agents.tools.surrealist_tools import CutUpProcessor


class FakeState:
    def __init__(self):
        self.thread_id = "t1"
        self.active_agents = ["red", "blue"]
        # Provide content attributes for agents
        self.red_content = {"notes": "word " * 60}  # should produce multiple fragments
        self.blue_content = {"lines": ["alpha beta gamma delta epsilon zeta eta theta iota kappa"]}


def test_cut_up_processor_generates_fragments():
    proc = CutUpProcessor()
    state = FakeState()
    out = proc(state)
    assert hasattr(out, "cut_up_fragments")
    frags = out.cut_up_fragments
    assert isinstance(frags, list)
    # Should generate at least one fragment
    assert len(frags) >= 1
    # Each fragment should be non-empty string
    assert all(isinstance(f, str) and f.strip() for f in frags)

