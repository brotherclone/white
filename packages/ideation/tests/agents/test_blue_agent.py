import datetime
from unittest.mock import patch

from white_core.artifacts.alternate_timeline_artifact import AlternateTimelineArtifact
from white_core.artifacts.quantum_tape_label_artifact import QuantumTapeLabelArtifact
from white_core.concepts.biographical_period import BiographicalPeriod
from white_core.concepts.divergence_point import DivergencePoint
from white_core.concepts.quantum_tape_instrumentation import (
    QuantumTapeInstrumentationConfig,
)
from white_core.concepts.quantum_tape_musical_parameters import (
    QuantumTapeMusicalParameters,
)
from white_core.concepts.quantum_tape_production_aesthetic import (
    QuantumTapeProductionAesthetic,
)
from white_core.enums.chain_artifact_type import ChainArtifactType
from white_core.manifests.song_proposal import SongProposalIteration
from white_ideation.agents.blue_agent import BlueAgent
from white_ideation.agents.states.blue_agent_state import BlueAgentState


def test_generate_alternate_song_spec_mock():
    agent = BlueAgent()
    state = BlueAgentState()
    state.thread_id = "test_thread"
    result_state = agent.generate_alternate_song_spec(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)


def test_generate_tape_label_mock(monkeypatch, tmp_path):
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setenv("BLOCK_MODE", "false")
    monkeypatch.setenv("AGENT_MOCK_DATA_PATH", "tests/mocks")
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", str(tmp_path))

    agent = BlueAgent()
    state = BlueAgentState()
    state.thread_id = "test_thread"
    result_state = agent.generate_tape_label(state)

    tape_labels = [
        a for a in result_state.artifacts if isinstance(a, QuantumTapeLabelArtifact)
    ]
    assert len(tape_labels) == 1
    assert tape_labels[0].chain_artifact_type == ChainArtifactType.QUANTUM_TAPE_LABEL
    assert result_state.tape_label is not None


def _fake_alternate_history() -> AlternateTimelineArtifact:
    return AlternateTimelineArtifact(
        thread_id="test_thread",
        period=BiographicalPeriod(
            start_date=datetime.date(1978, 1, 1),
            end_date=datetime.date(1978, 6, 1),
            age_range=(21, 21),
            description="Test period",
        ),
        title="Test Alternate Timeline",
        divergence_point=DivergencePoint(
            when="After a test event",
            what_changed="A test change",
            why_plausible="Because this is a test",
        ),
    )


def _fake_musical_params() -> QuantumTapeMusicalParameters:
    return QuantumTapeMusicalParameters(
        bpm=110,
        key="G_major",
        mood="melancholy_folk_rock",
        instrumentation=QuantumTapeInstrumentationConfig(color=["mellotron"]),
        production_aesthetic=QuantumTapeProductionAesthetic(),
        lyrical_themes=["memory", "loss"],
    )


def test_generate_alternate_song_spec_includes_sounds_like(monkeypatch):
    """Regression for add-ideation-sounds-like: Blue Agent already computes
    musical_params.reference_artists via get_sounds_like_by_color() +
    sample_reference_artists() at extract_musical_parameters time, but the
    field was never actually read anywhere — computed and discarded."""
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("BLOCK_MODE", "false")

    agent = BlueAgent()
    state = BlueAgentState()
    state.thread_id = "test_thread"
    state.alternate_history = _fake_alternate_history()
    state.musical_params = QuantumTapeMusicalParameters(
        bpm=110,
        key="G_major",
        mood="melancholy_folk_rock",
        instrumentation=QuantumTapeInstrumentationConfig(color=["mellotron"]),
        production_aesthetic=QuantumTapeProductionAesthetic(),
        lyrical_themes=["memory", "loss"],
        reference_artists=["Test Reference Artist"],
    )

    counter_proposal = SongProposalIteration(
        iteration_id="blue_v1",
        bpm=110,
        tempo="3/4",
        key="G major",
        rainbow_color="blue",
        title="Test Blue Song",
        mood=["melancholic"],
        genres=["folk rock"],
        concept="A test concept long enough to pass the minimum length validator "
        "for the concept field, exploring erased timelines and loss.",
    )
    seen_prompts = []

    def fake_invoke_structured(llm, schema, prompt):
        seen_prompts.append(prompt)
        return counter_proposal

    with patch.object(agent, "_invoke_structured", side_effect=fake_invoke_structured):
        agent.generate_alternate_song_spec(state)

    assert len(seen_prompts) == 1
    assert "Test Reference Artist" in seen_prompts[0]


def test_generate_alternate_song_spec_includes_negative_constraints(monkeypatch):
    """Regression for add-ideation-negative-constraints: Blue's own
    counter-proposal prompt must honor negative_constraints, not just
    White's initial proposal and final rewrite."""
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("BLOCK_MODE", "false")

    agent = BlueAgent()
    state = BlueAgentState()
    state.thread_id = "test_thread"
    state.negative_constraints = "AVOID: the word 'tape', keys already used: G major"
    state.alternate_history = _fake_alternate_history()
    state.musical_params = _fake_musical_params()

    counter_proposal = SongProposalIteration(
        iteration_id="blue_v1",
        bpm=110,
        tempo="3/4",
        key="G major",
        rainbow_color="blue",
        title="Test Blue Song",
        mood=["melancholic"],
        genres=["folk rock"],
        concept="A test concept long enough to pass the minimum length validator "
        "for the concept field, exploring erased timelines and loss.",
    )
    seen_prompts = []

    def fake_invoke_structured(llm, schema, prompt):
        seen_prompts.append(prompt)
        return counter_proposal

    with patch.object(agent, "_invoke_structured", side_effect=fake_invoke_structured):
        result_state = agent.generate_alternate_song_spec(state)

    assert result_state.counter_proposal is not None
    assert len(seen_prompts) == 1
    assert state.negative_constraints in seen_prompts[0]
