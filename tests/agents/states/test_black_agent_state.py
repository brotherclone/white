import pytest

from app.agents.enums.chain_artifact_file_type import ChainArtifactFileType
from app.agents.models.audio_chain_artifact_file import AudioChainArtifactFile
from app.agents.models.text_chain_artifact_file import TextChainArtifactFile
from app.agents.states.black_agent_state import BlackAgentState
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.manifests.song_proposal import SongProposal, SongProposalIteration
from app.agents.models.evp_artifact import EVPArtifact
from app.agents.models.sigil_artifact import SigilArtifact
from app.agents.enums.sigil_type import SigilType
from app.agents.enums.sigil_state import SigilState


def test_black_agent_state_defaults():
    state = BlackAgentState()
    assert state.thread_id.startswith("black_thread_")
    assert state.white_proposal is None
    assert state.song_proposals is None
    assert state.counter_proposal is None
    assert isinstance(state.artifacts, list)
    assert state.artifacts == []
    assert state.human_instructions == ""
    assert isinstance(state.pending_human_tasks, list)
    assert state.pending_human_tasks == []
    assert state.awaiting_human_action is False


def test_black_agent_state_custom_fields():
    proposal_iter = SongProposalIteration(
        iteration_id="123",
        bpm=120,
        tempo="4/4",
        key="C Major",
        rainbow_color=the_rainbow_table_colors['Z'],
        title="Test",
        mood=[],
        genres=[],
        concept="Test concept"
    )
    proposal = SongProposal()
    audio_file = AudioChainArtifactFile(
        sample_rate=44100,
        duration=1.0,
        channels=1,
        base_path="",
        chain_artifact_file_type=ChainArtifactFileType.AUDIO,
    )
    transcript_file = TextChainArtifactFile(
        text="Test transcript",
        base_path="",
        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
    )
    evp = EVPArtifact(
        chain_artifact_type="evp",
        files=[audio_file],
        audio_segments=[audio_file],
        transcript=transcript_file,
        audio_mosiac=audio_file,
        noise_blended_audio=audio_file,
        thread_id="123",
    )
    sigil = SigilArtifact(
        thread_id="sigil-thread-1",
        wish="Manifest creative energy",
        statement_of_intent="To focus intent on project success",
        sigil_type=SigilType.WORD_METHOD,
        glyph_description="A stylized glyph combining project initials",
        activation_state=SigilState.CREATED,
        charging_instructions="Charge during full moon",
        files=[],
    )
    artifacts = [evp, sigil]
    pending_tasks = [{"task": "review"}]
    state = BlackAgentState(
        white_proposal=proposal_iter,
        song_proposals=proposal,
        counter_proposal=proposal_iter,
        artifacts=artifacts,
        human_instructions="Do something",
        pending_human_tasks=pending_tasks,
        awaiting_human_action=True
    )
    assert state.white_proposal is proposal_iter
    assert state.song_proposals is proposal
    assert state.counter_proposal is proposal_iter
    assert state.artifacts == artifacts
    assert state.human_instructions == "Do something"
    assert state.pending_human_tasks == pending_tasks
    assert state.awaiting_human_action is True
