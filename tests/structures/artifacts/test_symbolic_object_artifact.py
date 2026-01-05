from app.structures.artifacts.symbolic_object_artifact import SymbolicObjectArtifact
from app.structures.enums.chain_artifact_type import ChainArtifactType
from app.structures.enums.symbolic_object_category import SymbolicObjectCategory


def test_symbolic_object_defaults():
    artifact = SymbolicObjectArtifact(thread_id="thread-1")
    assert artifact.thread_id == "thread-1"
    assert (
        artifact.symbolic_object_category
        == SymbolicObjectCategory.INFORMATION_ARTIFACTS
    )
    assert artifact.name is None
    assert artifact.description is None


def test_symbolic_object_with_fields():
    artifact = SymbolicObjectArtifact(
        thread_id="thread-2",
        symbolic_object_category=SymbolicObjectCategory.CIRCULAR_TIME,
        name="Mock Clock",
        description="A looping temporal marker",
    )
    assert artifact.thread_id == "thread-2"
    assert artifact.symbolic_object_category == SymbolicObjectCategory.CIRCULAR_TIME
    assert artifact.name == "Mock Clock"
    assert artifact.description == "A looping temporal marker"


def test_for_prompt_basic():
    """Test for_prompt() with name and description."""
    artifact = SymbolicObjectArtifact(
        thread_id="test",
        name="The Eternal Clock",
        description="A timepiece that measures circular time, always returning to its starting point",
    )

    prompt = artifact.for_prompt()

    assert "Symbolic Object: The Eternal Clock" in prompt
    assert "A timepiece that measures circular time" in prompt
    assert "Category: information_artifacts" in prompt  # default


def test_for_prompt_all_fields():
    """Test for_prompt() with all fields."""
    artifact = SymbolicObjectArtifact(
        thread_id="complete",
        symbolic_object_category=SymbolicObjectCategory.CIRCULAR_TIME,
        name="Nash's 182 BPM Clock",
        description="A metronome-clock hybrid that tracks time in beats rather than seconds, perpetually cycling through the same 182 beats per minute",
    )

    prompt = artifact.for_prompt()

    assert "Symbolic Object: Nash's 182 BPM Clock" in prompt
    assert "metronome-clock hybrid" in prompt
    assert "Category: circular_time" in prompt


def test_for_prompt_minimal():
    """Test for_prompt() with minimal fields."""
    artifact = SymbolicObjectArtifact(thread_id="minimal")
    prompt = artifact.for_prompt()

    # With no name, description, should just show category
    assert "Category: information_artifacts" in prompt


def test_for_prompt_only_name():
    """Test for_prompt() with only name."""
    artifact = SymbolicObjectArtifact(thread_id="name_only", name="The Gateway")

    prompt = artifact.for_prompt()

    assert "Symbolic Object: The Gateway" in prompt
    assert "Category:" in prompt


def test_for_prompt_only_description():
    """Test for_prompt() with only description."""
    artifact = SymbolicObjectArtifact(
        thread_id="desc_only",
        description="A mysterious doorway that appears only at twilight",
    )

    prompt = artifact.for_prompt()

    assert "A mysterious doorway" in prompt
    assert "Category:" in prompt


def test_all_symbolic_categories():
    """Test creating object with each category."""
    for category in SymbolicObjectCategory:
        artifact = SymbolicObjectArtifact(
            thread_id="category_test", symbolic_object_category=category
        )
        assert artifact.symbolic_object_category == category


def test_circular_time_object():
    """Test creating a circular time object."""
    artifact = SymbolicObjectArtifact(
        thread_id="clock",
        symbolic_object_category=SymbolicObjectCategory.CIRCULAR_TIME,
        name="The Ouroboros Watch",
        description="A watch whose hands move in an endless loop, consuming their own path",
    )

    assert artifact.symbolic_object_category == SymbolicObjectCategory.CIRCULAR_TIME
    assert "Ouroboros" in artifact.name
    assert "endless loop" in artifact.description


def test_information_artifacts_object():
    """Test creating an information artifact object."""
    artifact = SymbolicObjectArtifact(
        thread_id="newspaper",
        symbolic_object_category=SymbolicObjectCategory.INFORMATION_ARTIFACTS,
        name="The EMORY Transmission",
        description="A recurring broadcast signal containing fragmented memories from alternative timelines",
    )

    assert (
        artifact.symbolic_object_category
        == SymbolicObjectCategory.INFORMATION_ARTIFACTS
    )
    assert "EMORY" in artifact.name


def test_liminal_objects():
    """Test creating a liminal object."""
    artifact = SymbolicObjectArtifact(
        thread_id="gateway",
        symbolic_object_category=SymbolicObjectCategory.LIMINAL_OBJECTS,
        name="Pine Barrens Gateway",
        description="A threshold between worlds hidden deep in the forest, only visible to those who know how to look",
    )

    assert artifact.symbolic_object_category == SymbolicObjectCategory.LIMINAL_OBJECTS
    assert "Gateway" in artifact.name
    assert "threshold" in artifact.description


def test_psychogeographic_object():
    """Test creating a psychogeographic object."""
    artifact = SymbolicObjectArtifact(
        thread_id="map",
        symbolic_object_category=SymbolicObjectCategory.PSYCHOGEOGRAPHIC,
        name="The Folding Map",
        description="A map that shows not physical locations but emotional resonances and temporal overlaps",
    )

    assert artifact.symbolic_object_category == SymbolicObjectCategory.PSYCHOGEOGRAPHIC
    assert "Map" in artifact.name


def test_flatten_all_fields():
    """Test flatten() includes all fields."""
    artifact = SymbolicObjectArtifact(
        thread_id="flatten_test",
        symbolic_object_category=SymbolicObjectCategory.LIMINAL_OBJECTS,
        name="Test Object",
        description="Test Description",
    )

    flat = artifact.flatten()

    assert flat["thread_id"] == "flatten_test"
    assert (
        flat["symbolic_object_category"] == SymbolicObjectCategory.LIMINAL_OBJECTS.value
    )
    assert flat["name"] == "Test Object"
    assert flat["description"] == "Test Description"
    assert flat["chain_artifact_type"] == ChainArtifactType.SYMBOLIC_OBJECT.value


def test_flatten_with_defaults():
    """Test flatten() with default values."""
    artifact = SymbolicObjectArtifact(thread_id="defaults")

    flat = artifact.flatten()

    assert flat["thread_id"] == "defaults"
    assert (
        flat["symbolic_object_category"]
        == SymbolicObjectCategory.INFORMATION_ARTIFACTS.value
    )
    assert flat["name"] is None
    assert flat["description"] is None


def test_chain_artifact_type_is_symbolic_object():
    """Test that chain_artifact_type is always SYMBOLIC_OBJECT."""
    artifact = SymbolicObjectArtifact(thread_id="type_test")
    assert artifact.chain_artifact_type == ChainArtifactType.SYMBOLIC_OBJECT


def test_default_category_is_information_artifacts():
    """Test that default category is INFORMATION_ARTIFACTS."""
    artifact = SymbolicObjectArtifact(thread_id="test")
    assert (
        artifact.symbolic_object_category
        == SymbolicObjectCategory.INFORMATION_ARTIFACTS
    )


def test_long_description():
    """Test with a long, detailed description."""
    long_desc = """A complex symbolic object that exists in multiple states simultaneously.
    It appears differently to each observer, reflecting their own relationship with time and memory.
    Some see it as a clock, others as a calendar, still others as a living organism.
    Its true nature remains unknowable, perhaps even to itself."""

    artifact = SymbolicObjectArtifact(
        thread_id="long_desc", name="The Multifaceted Observer", description=long_desc
    )

    assert artifact.description == long_desc
    assert "multiple states" in artifact.description


def test_special_characters_in_name():
    """Test name with special characters."""
    artifact = SymbolicObjectArtifact(
        thread_id="special",
        name="The ∞-Clock [Mark II]",
        description="A clock marked with the infinity symbol",
    )

    assert "∞" in artifact.name
    assert "[Mark II]" in artifact.name


def test_empty_string_vs_none():
    """Test difference between empty string and None."""
    artifact_none = SymbolicObjectArtifact(
        thread_id="none", name=None, description=None
    )

    artifact_empty = SymbolicObjectArtifact(thread_id="empty", name="", description="")

    assert artifact_none.name is None
    assert artifact_none.description is None
    assert artifact_empty.name == ""
    assert artifact_empty.description == ""


def test_update_category():
    """Test updating the category after creation."""
    artifact = SymbolicObjectArtifact(
        thread_id="update",
        symbolic_object_category=SymbolicObjectCategory.CIRCULAR_TIME,
    )

    assert artifact.symbolic_object_category == SymbolicObjectCategory.CIRCULAR_TIME

    artifact.symbolic_object_category = SymbolicObjectCategory.LIMINAL_OBJECTS
    assert artifact.symbolic_object_category == SymbolicObjectCategory.LIMINAL_OBJECTS
