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


def test_minimal_sigil_artifact():
    """Test creating sigil with only thread_id (all fields optional)."""
    artifact = SigilArtifact(thread_id="minimal")

    assert artifact.thread_id == "minimal"
    assert artifact.chain_artifact_type == ChainArtifactType.SIGIL

    # Check all optional fields are None or default
    assert artifact.wish is None
    assert artifact.statement_of_intent is None
    assert artifact.sigil_type is None
    assert artifact.glyph_description is None
    assert artifact.glyph_components == []
    assert artifact.activation_state is None
    assert artifact.charging_instructions is None


def test_for_prompt_basic():
    """Test for_prompt() with basic fields."""
    artifact = SigilArtifact(
        thread_id="test",
        wish="To find clarity",
        statement_of_intent="I will achieve mental clarity",
    )

    prompt = artifact.for_prompt()

    assert "Wish: To find clarity" in prompt
    assert "Statement of Intent: I will achieve mental clarity" in prompt


def test_for_prompt_with_all_fields():
    """Test for_prompt() with all fields populated."""
    artifact = SigilArtifact(
        thread_id="complete",
        wish="To manifest success",
        statement_of_intent="My will shapes reality",
        sigil_type=SigilType.PICTORIAL,
        glyph_description="Interlocking spiral with three points",
        glyph_components=["spiral", "triangle", "circle"],
        activation_state=SigilState.CHARGED,
        charging_instructions="Focus on the sigil during meditation for 7 nights",
    )

    prompt = artifact.for_prompt()

    assert "Wish: To manifest success" in prompt
    assert "Statement of Intent: My will shapes reality" in prompt
    assert "Sigil Type: pictorial" in prompt
    assert "Glyph Description: Interlocking spiral with three points" in prompt
    assert "Glyph Components: spiral, triangle, circle" in prompt
    assert "Activation State: charged" in prompt
    assert (
        "Charging Instructions: Focus on the sigil during meditation for 7 nights"
        in prompt
    )


def test_for_prompt_empty():
    """Test for_prompt() with no optional fields returns empty string."""
    artifact = SigilArtifact(thread_id="empty")
    prompt = artifact.for_prompt()
    assert prompt == ""


def test_for_prompt_partial_fields():
    """Test for_prompt() with some fields populated."""
    artifact = SigilArtifact(
        thread_id="partial",
        wish="Test wish",
        sigil_type=SigilType.MANTRIC,
        activation_state=SigilState.CHARGING,
    )

    prompt = artifact.for_prompt()

    assert "Wish: Test wish" in prompt
    assert "Sigil Type: mantric" in prompt
    assert "Activation State: charging" in prompt
    assert "Statement of Intent:" not in prompt
    assert "Glyph Description:" not in prompt


def test_all_sigil_types():
    """Test creating sigil with each type."""
    for sigil_type in SigilType:
        artifact = SigilArtifact(thread_id="type_test", sigil_type=sigil_type)
        assert artifact.sigil_type == sigil_type


def test_all_sigil_states():
    """Test creating sigil with each activation state."""
    for state in SigilState:
        artifact = SigilArtifact(thread_id="state_test", activation_state=state)
        assert artifact.activation_state == state


def test_flatten_all_fields():
    """Test flatten() includes all fields."""
    artifact = SigilArtifact(
        thread_id="flatten_test",
        wish="Flatten wish",
        statement_of_intent="Flatten intent",
        sigil_type=SigilType.WORD_METHOD,
        glyph_description="Description",
        glyph_components=["a", "b", "c"],
        activation_state=SigilState.BURIED,
        charging_instructions="Bury under oak tree",
    )

    flat = artifact.flatten()

    assert flat["thread_id"] == "flatten_test"
    assert flat["wish"] == "Flatten wish"
    assert flat["statement_of_intent"] == "Flatten intent"
    assert flat["sigil_type"] == SigilType.WORD_METHOD.value
    assert flat["glyph_description"] == "Description"
    assert flat["glyph_components"] == ["a", "b", "c"]
    assert flat["activation_state"] == SigilState.BURIED.value
    assert flat["charging_instructions"] == "Bury under oak tree"
    assert flat["chain_artifact_type"] == ChainArtifactType.SIGIL


def test_flatten_with_none_enums():
    """Test flatten() handles None enum values correctly."""
    artifact = SigilArtifact(
        thread_id="none_enums", wish="Test", sigil_type=None, activation_state=None
    )

    flat = artifact.flatten()

    assert flat["sigil_type"] is None
    assert flat["activation_state"] is None


def test_glyph_components_default_empty_list():
    """Test glyph_components defaults to empty list."""
    artifact = SigilArtifact(thread_id="test")
    assert artifact.glyph_components == []
    assert isinstance(artifact.glyph_components, list)


def test_complex_glyph_components():
    """Test with many glyph components."""
    components = [
        "circle",
        "square",
        "triangle",
        "pentagram",
        "hexagram",
        "spiral",
        "cross",
    ]
    artifact = SigilArtifact(thread_id="complex", glyph_components=components)
    assert len(artifact.glyph_components) == 7
    assert all(comp in artifact.glyph_components for comp in components)


def test_sigil_lifecycle():
    """Test sigil through different activation states."""
    artifact = SigilArtifact(
        thread_id="lifecycle",
        wish="Transform",
        statement_of_intent="I will transform",
        sigil_type=SigilType.ALPHABET_OF_DESIRE,
    )

    # Initial state
    assert artifact.activation_state is None

    # Update through lifecycle
    states = [
        SigilState.CREATED,
        SigilState.AWAITING_CHARGE,
        SigilState.CHARGING,
        SigilState.CHARGED,
        SigilState.BURIED,
    ]

    for state in states:
        artifact.activation_state = state
        assert artifact.activation_state == state


def test_charging_instructions_field():
    """Test charging_instructions field."""
    instructions = "Meditate on the sigil at midnight for three consecutive nights"
    artifact = SigilArtifact(thread_id="charging", charging_instructions=instructions)
    assert artifact.charging_instructions == instructions


def test_chain_artifact_type_is_sigil():
    """Test that chain_artifact_type is always SIGIL."""
    artifact = SigilArtifact(thread_id="type_test")
    assert artifact.chain_artifact_type == ChainArtifactType.SIGIL


def test_word_method_sigil():
    """Test creating a word method sigil."""
    artifact = SigilArtifact(
        thread_id="word_method",
        wish="Success in endeavors",
        statement_of_intent="MY WILL BRINGS SUCCESS",
        sigil_type=SigilType.WORD_METHOD,
        glyph_components=["M", "Y", "W", "L", "B", "R", "N", "G", "S", "U", "C"],
        activation_state=SigilState.CREATED,
    )

    assert artifact.sigil_type == SigilType.WORD_METHOD
    assert len(artifact.glyph_components) == 11  # Unique letters


def test_pictorial_sigil():
    """Test creating a pictorial sigil."""
    artifact = SigilArtifact(
        thread_id="pictorial",
        wish="Protection",
        sigil_type=SigilType.PICTORIAL,
        glyph_description="An eye within a pentagram, surrounded by protective runes",
        glyph_components=["eye", "pentagram", "runes"],
        activation_state=SigilState.AWAITING_CHARGE,
    )

    assert artifact.sigil_type == SigilType.PICTORIAL
    assert "eye within a pentagram" in artifact.glyph_description


def test_mantric_sigil():
    """Test creating a mantric sigil."""
    artifact = SigilArtifact(
        thread_id="mantric",
        statement_of_intent="AUM MANI PADME HUM",
        sigil_type=SigilType.MANTRIC,
        activation_state=SigilState.CHARGING,
        charging_instructions="Chant 108 times daily",
    )

    assert artifact.sigil_type == SigilType.MANTRIC
    assert "108 times" in artifact.charging_instructions
