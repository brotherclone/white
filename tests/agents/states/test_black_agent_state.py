from app.agents.states.black_agent_state import BlackAgentState
from app.structures.artifacts.audio_artifact_file import AudioChainArtifactFile
from app.structures.artifacts.evp_artifact import EVPArtifact
from app.structures.artifacts.sigil_artifact import SigilArtifact
from app.structures.artifacts.text_artifact_file import TextChainArtifactFile
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
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
    assert state.human_instructions == ""
    assert isinstance(state.pending_human_tasks, list)
    assert state.pending_human_tasks == []
    assert state.awaiting_human_action is False
    assert state.should_update_proposal_with_evp is False
    assert state.should_update_proposal_with_sigil is False


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
    audio_file = AudioChainArtifactFile(
        sample_rate=44100,
        duration=1.0,
        channels=1,
        base_path="",
        chain_artifact_file_type=ChainArtifactFileType.AUDIO,
        artifact_name="test_audio",
        artifact_id="123",
        thread_id="345",
        rainbow_color=the_rainbow_table_colors["Z"],
        file_name="test_audio.wav",
    )
    transcript_file = TextChainArtifactFile(
        text_content="Test transcript",
        base_path="",
        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
        artifact_name="test_md",
        artifact_id="123",
        thread_id="345",
        rainbow_color=the_rainbow_table_colors["Z"],
        file_name="test_transcript.md",
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
        artifact_report=TextChainArtifactFile(
            text_content="Sigil created successfully.",
            base_path="",
            chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
            artifact_name="sigil_report",
            artifact_id="sigil-001",
            thread_id="sigil-thread-1",
            rainbow_color=the_rainbow_table_colors["Z"],
            file_name="sigil_report.md",
        ),
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
        awaiting_human_action=True,
        should_update_proposal_with_evp=True,
        should_update_proposal_with_sigil=True,
    )
    assert state.white_proposal is proposal_iter
    assert state.song_proposals is proposal
    assert state.counter_proposal is proposal_iter
    assert state.artifacts == artifacts
    assert state.human_instructions == "Do something"
    assert state.pending_human_tasks == pending_tasks
    assert state.awaiting_human_action is True
    assert state.should_update_proposal_with_evp is True
    assert state.should_update_proposal_with_sigil is True
