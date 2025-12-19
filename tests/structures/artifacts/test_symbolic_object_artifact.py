from app.structures.artifacts.symbolic_object_artifact import SymbolicObjectArtifact
from app.structures.enums.symbolic_object_category import SymbolicObjectCategory

# ToDo: Add for_prompt() tests
# ToDo: Skimpy!


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
