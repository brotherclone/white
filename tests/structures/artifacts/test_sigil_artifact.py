from app.structures.artifacts.sigil_artifact import SigilArtifact
from app.structures.enums.chain_artifact_type import ChainArtifactType
from app.structures.enums.sigil_state import SigilState
from app.structures.enums.sigil_type import SigilType


def test_sigil_artifact():
    """Test basic SigilArtifact creation"""
    artifact = SigilArtifact(
        thread_id="test-thread", wish="Test wish", statement_of_intent="I will succeed"
    )
    assert artifact.thread_id == "test-thread"
    assert artifact.wish == "Test wish"
    assert artifact.statement_of_intent == "I will succeed"
    assert artifact.chain_artifact_type == ChainArtifactType.SIGIL


def test_sigil_artifact_with_enums():
    """Test SigilArtifact with enum fields"""
    sigil_type = list(SigilType)[0]
    sigil_state = list(SigilState)[0]

    artifact = SigilArtifact(
        thread_id="test-thread", sigil_type=sigil_type, activation_state=sigil_state
    )
    assert artifact.sigil_type == sigil_type
    assert artifact.activation_state == sigil_state


def test_sigil_artifact_with_glyph_components():
    """Test SigilArtifact with glyph components"""
    artifact = SigilArtifact(
        thread_id="test-thread",
        glyph_description="A circular pattern",
        glyph_components=["circle", "line", "dot"],
    )
    assert artifact.glyph_description == "A circular pattern"
    assert len(artifact.glyph_components) == 3
    assert "circle" in artifact.glyph_components
