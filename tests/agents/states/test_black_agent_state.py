from pathlib import Path
from app.agents.states.black_agent_state import BlackAgentState
from app.structures.artifacts.audio_artifact_file import AudioChainArtifactFile
from app.structures.artifacts.evp_artifact import EVPArtifact
from app.structures.artifacts.sigil_artifact import SigilArtifact
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType
from app.structures.enums.sigil_state import SigilState
from app.structures.enums.sigil_type import SigilType
from app.structures.manifests.song_proposal import SongProposal, SongProposalIteration


def test_black_agent_state_defaults():
    state = BlackAgentState()
    assert state.thread_id is None
    assert state.white_proposal is None
    assert state.song_proposals is None
    assert state.counter_proposal is None
    assert isinstance(state.artifacts, list)
    assert state.artifacts == []
    assert state.should_update_proposal_with_evp is False


def test_black_agent_state_custom_fields():
    proposal_iter = SongProposalIteration(
        iteration_id="file_1",
        bpm=120,
        tempo="4/4",
        key="C Major",
        rainbow_color=the_rainbow_table_colors["Z"],
        title="Test",
        mood=["testy"],
        genres=["test-rock"],
        concept="Test concept in which a song proposal iteration is created with mock values for testing. Now, if I have to sit here and type, oh good it's over 100 now.",
    )
    proposal = SongProposal()

    # Load mock audio file as bytes
    mock_audio_path = Path(__file__).parent.parent.parent / "mocks" / "mock.wav"
    with open(mock_audio_path, "rb") as f:
        audio_bytes = f.read()

    audio_file = AudioChainArtifactFile(
        sample_rate=44100,
        duration=5.0,
        channels=2,
        audio_bytes=audio_bytes,
        base_path="/tmp/artifacts/",
        artifact_name="test_audio",
        thread_id="345",
    )

    evp = EVPArtifact(
        chain_artifact_type=ChainArtifactType.EVP_ARTIFACT,
        chain_artifact_file_type=ChainArtifactFileType.YML,
        file_path="/tmp/artifacts/",
        file_name="test_evp.yml",
        audio_segments=[audio_file],
        transcript="hi",
        audio_mosiac=audio_file,
        noise_blended_audio=audio_file,
        thread_id="123",
    )
    sigil = SigilArtifact(
        chain_artifact_type=ChainArtifactType.SIGIL,
        chain_artifact_file_type=ChainArtifactFileType.YML,
        file_path="/tmp/artifacts/",
        file_name="test_sigil.yml",
        thread_id="sigil-thread-1",
        wish="Manifest creative energy",
        statement_of_intent="To focus intent on project success",
        sigil_type=SigilType.WORD_METHOD,
        glyph_description="A stylized glyph combining project initials",
        activation_state=SigilState.CREATED,
        charging_instructions="Charge during full moon",
    )
    artifacts = [evp, sigil]
    state = BlackAgentState(
        white_proposal=proposal_iter,
        song_proposals=proposal,
        counter_proposal=proposal_iter,
        artifacts=artifacts,
        should_update_proposal_with_evp=True,
    )
    assert state.white_proposal is proposal_iter
    assert state.song_proposals is proposal
    assert state.counter_proposal is proposal_iter
    assert state.artifacts == artifacts
    assert state.should_update_proposal_with_evp is True
