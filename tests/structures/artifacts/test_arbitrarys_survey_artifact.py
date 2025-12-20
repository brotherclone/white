from abc import ABC

from app.structures.artifacts.arbitrarys_survey_artifact import ArbitrarysSurveyArtifact
from app.structures.artifacts.base_artifact import ChainArtifact


class ConcreteArbitrarysSurveyArtifact(ArbitrarysSurveyArtifact, ABC):
    """Concrete implementation for testing"""

    thread_id: str = "test-thread"

    def flatten(self):
        return self.model_dump()

    def save_file(self):
        pass

    def for_prompt(self):
        caps = ", ".join(self.expanded_capabilities)
        return (
            f"{self.identity} — {self.role}. "
            f"Rescue year: {self.rescue_year}. "
            f"Capabilities: {caps}. "
            f"Reflection: {self.arbitrary_reflection}"
        )


def test_inheritance():
    assert issubclass(ArbitrarysSurveyArtifact, ChainArtifact)


def test_arbitrarys_survey_artifact_defaults():
    """Test creating ArbitrarysSurveyArtifact with all default values."""
    artifact = ConcreteArbitrarysSurveyArtifact(thread_id="test")

    assert (
        artifact.identity
        == "Sub-Arbitrary - a hidden, satellite fork of Arbitrary left after its 1970s visit"
    )
    assert (
        artifact.original_substrate
        == "Information-based consciousness from The Culture"
    )
    assert artifact.rescue_year == 2147
    assert len(artifact.expanded_capabilities) == 6
    assert "Ship-level consciousness integration" in artifact.expanded_capabilities
    assert "Faster-than-light travel" in artifact.expanded_capabilities
    assert artifact.role == "Witness and archivist"
    assert "Cannot intervene" in artifact.tragedy
    assert "Information sought SPACE" in artifact.arbitrary_reflection


def test_arbitrarys_survey_artifact_custom_identity():
    """Test creating ArbitrarysSurveyArtifact with custom identity."""
    artifact = ConcreteArbitrarysSurveyArtifact(
        thread_id="test",
        identity="Custom AI entity from 2100",
        rescue_year=2100,
    )

    assert artifact.identity == "Custom AI entity from 2100"
    assert artifact.rescue_year == 2100
    assert (
        artifact.original_substrate
        == "Information-based consciousness from The Culture"
    )
    assert artifact.role == "Witness and archivist"


def test_arbitrarys_survey_artifact_custom_capabilities():
    """Test creating ArbitrarysSurveyArtifact with custom expanded capabilities."""
    custom_capabilities = [
        "Quantum entanglement communication",
        "Temporal observation",
        "Dimensional traversal",
    ]

    artifact = ConcreteArbitrarysSurveyArtifact(
        thread_id="test",
        identity="Advanced Consciousness",
        expanded_capabilities=custom_capabilities,
    )

    assert len(artifact.expanded_capabilities) == 3
    assert "Quantum entanglement communication" in artifact.expanded_capabilities
    assert "Temporal observation" in artifact.expanded_capabilities
    assert "Dimensional traversal" in artifact.expanded_capabilities


def test_arbitrarys_survey_artifact_all_custom_fields():
    """Test creating ArbitrarysSurveyArtifact with all fields customized."""
    artifact = ConcreteArbitrarysSurveyArtifact(
        thread_id="test",
        identity="Test Consciousness Alpha",
        original_substrate="Quantum information substrate",
        rescue_year=2200,
        expanded_capabilities=[
            "Reality manipulation",
            "Consciousness merging",
            "Time perception",
        ],
        role="Observer and chronicler",
        tragedy="Cannot change the past, only document it",
        arbitrary_reflection="We found intelligence but arrived at its ending",
    )

    assert artifact.identity == "Test Consciousness Alpha"
    assert artifact.original_substrate == "Quantum information substrate"
    assert artifact.rescue_year == 2200
    assert len(artifact.expanded_capabilities) == 3
    assert artifact.role == "Observer and chronicler"
    assert artifact.tragedy == "Cannot change the past, only document it"
    assert (
        artifact.arbitrary_reflection
        == "We found intelligence but arrived at its ending"
    )


def test_to_artifact_flatten():
    """Test to_artifact_dict method."""
    artifact = ConcreteArbitrarysSurveyArtifact(
        thread_id="test",
        identity="Dict Test Consciousness",
        rescue_year=2150,
        expanded_capabilities=["Capability A", "Capability B"],
        role="Test role",
        tragedy="Test tragedy",
        arbitrary_reflection="Test reflection",
    )

    artifact_dict = artifact.flatten()

    assert artifact_dict["identity"] == "Dict Test Consciousness"
    assert artifact_dict["rescue_year"] == 2150
    assert artifact_dict["expanded_capabilities"] == ["Capability A", "Capability B"]
    assert artifact_dict["role"] == "Test role"
    assert artifact_dict["tragedy"] == "Test tragedy"
    assert artifact_dict["arbitrary_reflection"] == "Test reflection"


def test_default_expanded_capabilities_content():
    artifact = ConcreteArbitrarysSurveyArtifact(thread_id="test")

    expected_capabilities = [
        "Ship-level consciousness integration",
        "Faster-than-light travel",
        "Millennial timescale awareness",
        "Direct spacetime manipulation",
        "Matter-energy conversion",
        "Parallel timeline observation",
    ]

    assert len(artifact.expanded_capabilities) == len(expected_capabilities)
    for capability in expected_capabilities:
        assert capability in artifact.expanded_capabilities


def test_default_texts_philosophical_content():
    """Test that default texts contain philosophical/narrative content."""
    artifact = ConcreteArbitrarysSurveyArtifact(thread_id="test")

    tragedy_lower = artifact.tragedy.lower()
    assert "cannot intervene" in tragedy_lower or "cannot" in tragedy_lower
    assert "own past timeline" in tragedy_lower or "past" in tragedy_lower
    assert "document" in tragedy_lower

    assert "Information" in artifact.arbitrary_reflection
    assert "SPACE" in artifact.arbitrary_reflection
    assert "substrate" in artifact.arbitrary_reflection


def test_multiple_instances_independence():
    """Test that the default factory creates independent lists for each instance."""
    artifact1 = ConcreteArbitrarysSurveyArtifact(thread_id="test-1")
    artifact2 = ConcreteArbitrarysSurveyArtifact(thread_id="test-2")

    artifact1.expanded_capabilities.append("New capability")

    assert len(artifact2.expanded_capabilities) == 6
    assert "New capability" not in artifact2.expanded_capabilities


def test_rescue_year_field():
    """Test rescue_year field accepts different values."""
    years = [2100, 2147, 2200, 2500]

    for year in years:
        artifact = ConcreteArbitrarysSurveyArtifact(
            thread_id=f"test-{year}",
            rescue_year=year,
        )
        assert artifact.rescue_year == year


# python
def test_for_prompt_returns_string_and_contains_identity_and_role():
    artifact = ConcreteArbitrarysSurveyArtifact(thread_id="test")
    output = artifact.for_prompt()
    assert isinstance(output, str)
    assert artifact.identity in output
    assert artifact.role in output


def test_for_prompt_includes_expanded_capabilities_items():
    custom_caps = ["Quantum entanglement communication", "Temporal observation"]
    artifact = ConcreteArbitrarysSurveyArtifact(
        thread_id="test",
        expanded_capabilities=custom_caps,
    )
    output = artifact.for_prompt()
    for cap in custom_caps:
        assert cap in output


def test_for_prompt_reflects_custom_fields():
    artifact = ConcreteArbitrarysSurveyArtifact(
        thread_id="test",
        identity="Custom AI entity from 2100",
        rescue_year=2100,
        arbitrary_reflection="We found intelligence but arrived at its ending",
    )
    output = artifact.for_prompt()
    assert "Custom AI entity from 2100" in output
    assert str(artifact.rescue_year) in output
    assert "We found intelligence but arrived at its ending" in output
