from pydantic import ValidationError
import pytest

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.artifacts.last_human_artifact import LastHumanArtifact
from app.structures.enums.last_human_vulnerability_type import (
    LastHumanVulnerabilityType,
)
from app.structures.enums.last_human_documentation_type import (
    LastHumanDocumentationType,
)


class ConcreteLastHumanArtifact(LastHumanArtifact):
    """Concrete implementation for testing"""

    thread_id: str = "test-thread"

    def flatten(self):
        return self.model_dump()

    def save_file(self):
        pass


def test_inheritance():
    assert issubclass(LastHumanArtifact, ChainArtifact)


def test_last_human_artifact_required_fields():
    """Test creating LastHumanArtifact with required fields only."""
    artifact = ConcreteLastHumanArtifact(
        thread_id="test-thread",
        name="Maria Santos",
        age=42,
        location="Coastal Village, Philippines",
        year_documented=2045,
        parallel_vulnerability=LastHumanVulnerabilityType.DISPLACEMENT,
        vulnerability_details="Forced to leave ancestral fishing village due to sea level rise",
        environmental_stressor="Rising sea levels flooding village",
        documentation_type=LastHumanDocumentationType.DISPLACEMENT,
        last_days_scenario="Packing final belongings as waters rise around home",
    )

    assert artifact.name == "Maria Santos"
    assert artifact.age == 42
    assert artifact.pronouns == "they/them"
    assert artifact.location == "Coastal Village, Philippines"
    assert artifact.year_documented == 2045
    assert artifact.parallel_vulnerability == LastHumanVulnerabilityType.DISPLACEMENT
    assert artifact.documentation_type == LastHumanDocumentationType.DISPLACEMENT
    assert artifact.latitude is None
    assert artifact.longitude is None
    assert artifact.occupation is None
    assert artifact.family_situation is None
    assert artifact.daily_routine is None
    assert artifact.adaptation_attempts == []
    assert artifact.significant_object is None
    assert artifact.final_thought is None


def test_last_human_artifact_all_fields():
    """Test creating LastHumanArtifact with all fields populated."""
    artifact = ConcreteLastHumanArtifact(
        thread_id="test-thread",
        name="Ahmed Hassan",
        age=58,
        pronouns="he/him",
        location="Nile Delta, Egypt",
        latitude=31.0,
        longitude=31.5,
        year_documented=2050,
        parallel_vulnerability=LastHumanVulnerabilityType.RESOURCE_COLLAPSE,
        vulnerability_details="Agricultural collapse due to saltwater intrusion",
        occupation="Farmer, fourth generation",
        family_situation="Wife and three adult children, one grandchild",
        daily_routine="Wake at dawn, tend fields that no longer grow, fetch water from distant well",
        environmental_stressor="Saltwater intrusion destroying farmland",
        adaptation_attempts=[
            "Switched to salt-tolerant crops",
            "Dug deeper wells",
            "Tried aquaculture",
        ],
        documentation_type=LastHumanDocumentationType.RESILIENCE,
        last_days_scenario="Teaching grandson traditional songs while watching dead fields",
        significant_object="Family farming ledger, three generations of harvests",
        final_thought="Will anyone remember the taste of our dates?",
    )

    assert artifact.name == "Ahmed Hassan"
    assert artifact.age == 58
    assert artifact.pronouns == "he/him"
    assert artifact.latitude == 31.0
    assert artifact.longitude == 31.5
    assert artifact.occupation == "Farmer, fourth generation"
    assert artifact.family_situation == "Wife and three adult children, one grandchild"
    assert artifact.daily_routine is not None
    assert len(artifact.adaptation_attempts) == 3
    assert (
        artifact.significant_object
        == "Family farming ledger, three generations of harvests"
    )
    assert artifact.final_thought is not None


def test_age_validation():
    """Test age validation (0-120)."""
    # Valid ages
    artifact_young = ConcreteLastHumanArtifact(
        thread_id="test",
        name="Child",
        age=0,
        location="Test",
        year_documented=2040,
        parallel_vulnerability=LastHumanVulnerabilityType.HEALTH_CASCADE,
        vulnerability_details="Test",
        environmental_stressor="Test",
        documentation_type=LastHumanDocumentationType.WITNESS,
        last_days_scenario="Test",
    )
    assert artifact_young.age == 0

    artifact_old = ConcreteLastHumanArtifact(
        thread_id="test",
        name="Elder",
        age=120,
        location="Test",
        year_documented=2040,
        parallel_vulnerability=LastHumanVulnerabilityType.ISOLATION,
        vulnerability_details="Test",
        environmental_stressor="Test",
        documentation_type=LastHumanDocumentationType.WITNESS,
        last_days_scenario="Test",
    )
    assert artifact_old.age == 120

    # Invalid: negative
    with pytest.raises(ValidationError):
        ConcreteLastHumanArtifact(
            thread_id="test",
            name="Invalid",
            age=-1,
            location="Test",
            year_documented=2040,
            parallel_vulnerability=LastHumanVulnerabilityType.DISPLACEMENT,
            vulnerability_details="Test",
            environmental_stressor="Test",
            documentation_type=LastHumanDocumentationType.DEATH,
            last_days_scenario="Test",
        )

    # Invalid: too old
    with pytest.raises(ValidationError):
        ConcreteLastHumanArtifact(
            thread_id="test",
            name="Invalid",
            age=121,
            location="Test",
            year_documented=2040,
            parallel_vulnerability=LastHumanVulnerabilityType.DISPLACEMENT,
            vulnerability_details="Test",
            environmental_stressor="Test",
            documentation_type=LastHumanDocumentationType.DEATH,
            last_days_scenario="Test",
        )


def test_year_documented_validation():
    """Test year_documented validation (2020-2150)."""
    # Valid years
    artifact_2020 = ConcreteLastHumanArtifact(
        thread_id="test",
        name="Test 2020",
        age=30,
        location="Test",
        year_documented=2020,
        parallel_vulnerability=LastHumanVulnerabilityType.TOXIC_EXPOSURE,
        vulnerability_details="Test",
        environmental_stressor="Test",
        documentation_type=LastHumanDocumentationType.DEATH,
        last_days_scenario="Test",
    )
    assert artifact_2020.year_documented == 2020

    # Invalid: too early
    with pytest.raises(ValidationError):
        ConcreteLastHumanArtifact(
            thread_id="test",
            name="Invalid",
            age=30,
            location="Test",
            year_documented=2019,
            parallel_vulnerability=LastHumanVulnerabilityType.DISPLACEMENT,
            vulnerability_details="Test",
            environmental_stressor="Test",
            documentation_type=LastHumanDocumentationType.DEATH,
            last_days_scenario="Test",
        )


def test_to_artifact_dict():
    """Test to_artifact_dict method."""
    artifact = ConcreteLastHumanArtifact(
        thread_id="test",
        name="Test Person",
        age=35,
        location="Test City, Test Country",
        year_documented=2045,
        parallel_vulnerability=LastHumanVulnerabilityType.CLIMATE_REFUGEE,
        vulnerability_details="Climate-induced migration",
        occupation="Teacher",
        environmental_stressor="Prolonged drought",
        documentation_type=LastHumanDocumentationType.DISPLACEMENT,
        last_days_scenario="Final lesson to empty classroom",
        significant_object="Chalkboard eraser",
    )

    artifact_dict = artifact.to_artifact_dict()

    assert artifact_dict["name"] == "Test Person"
    assert artifact_dict["age"] == 35
    assert artifact_dict["location"] == "Test City, Test Country"
    assert artifact_dict["year"] == 2045
    assert artifact_dict["vulnerability"] == "climate_refugee"
    assert artifact_dict["vulnerability_details"] == "Climate-induced migration"
    assert artifact_dict["occupation"] == "Teacher"
    assert artifact_dict["environmental_stressor"] == "Prolonged drought"
    assert artifact_dict["documentation_type"] == "displacement"
    assert artifact_dict["scenario"] == "Final lesson to empty classroom"
    assert artifact_dict["significant_object"] == "Chalkboard eraser"


def test_summary_text():
    """Test summary_text method."""
    artifact = ConcreteLastHumanArtifact(
        thread_id="test",
        name="Elena Rodriguez",
        age=67,
        location="Amazon Basin, Brazil",
        year_documented=2055,
        parallel_vulnerability=LastHumanVulnerabilityType.ECONOMIC_EXTINCTION,
        vulnerability_details="Traditional way of life impossible after deforestation",
        occupation="Indigenous healer",
        environmental_stressor="Complete deforestation of ancestral territory",
        documentation_type=LastHumanDocumentationType.WITNESS,
        last_days_scenario="Recording oral history for future generations",
    )

    summary = artifact.summary_text()

    assert "Elena Rodriguez" in summary
    assert "67" in summary
    assert "Amazon Basin, Brazil" in summary
    assert "2055" in summary
    assert "Indigenous healer" in summary
    assert "Traditional way of life impossible after deforestation" in summary
    assert "Complete deforestation of ancestral territory" in summary


def test_summary_text_without_occupation():
    """Test summary_text when occupation is not provided."""
    artifact = ConcreteLastHumanArtifact(
        thread_id="test",
        name="Anonymous",
        age=25,
        location="Unnamed Village",
        year_documented=2040,
        parallel_vulnerability=LastHumanVulnerabilityType.ENTANGLEMENT,
        vulnerability_details="Caught in systemic collapse",
        environmental_stressor="Multiple cascading failures",
        documentation_type=LastHumanDocumentationType.DEATH,
        last_days_scenario="Final moments",
    )

    summary = artifact.summary_text()

    assert "Anonymous" in summary
    assert "Local resident" in summary
