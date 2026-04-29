from white_core.artifacts.quantum_tape_label_artifact import QuantumTapeLabelArtifact
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
