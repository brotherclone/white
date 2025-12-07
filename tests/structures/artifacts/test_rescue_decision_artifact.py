from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.artifacts.rescue_decision_artifact import RescueDecisionArtifact
from app.structures.artifacts.arbitrarys_survey_artifact import ArbitrarysSurveyArtifact
from app.structures.artifacts.last_human_artifact import LastHumanArtifact
from app.structures.artifacts.species_extinction_artifact import (
    SpeciesExtinctionArtifact,
)
from app.structures.enums.extinction_cause import ExtinctionCause
from app.structures.enums.last_human_vulnerability_type import (
    LastHumanVulnerabilityType,
)
from app.structures.enums.last_human_documentation_type import (
    LastHumanDocumentationType,
)


class ConcreteArbitrarysSurveyArtifact(ArbitrarysSurveyArtifact):
    """Concrete implementation for testing"""

    thread_id: str = "test-thread"

    def save_file(self):
        pass


class ConcreteLastHumanArtifact(LastHumanArtifact):
    """Concrete implementation for testing"""

    thread_id: str = "test-thread"

    def save_file(self):
        pass


class ConcreteSpeciesExtinctionArtifact(SpeciesExtinctionArtifact):
    """Concrete implementation for testing"""

    thread_id: str = "test-thread"

    def save_file(self):
        pass


class ConcreteRescueDecisionArtifact(RescueDecisionArtifact):
    """Concrete implementation for testing"""

    thread_id: str = "test-thread"

    def save_file(self):
        pass


def test_inheritance():
    assert issubclass(RescueDecisionArtifact, ChainArtifact)


def test_rescue_decision_artifact_creation():
    """Test creating RescueDecisionArtifact with required fields."""
    rescued = ConcreteArbitrarysSurveyArtifact(thread_id="test")

    human1 = ConcreteLastHumanArtifact(
        thread_id="test",
        name="Human One",
        age=45,
        location="Location A",
        year_documented=2045,
        parallel_vulnerability=LastHumanVulnerabilityType.DISPLACEMENT,
        vulnerability_details="Test",
        environmental_stressor="Test",
        documentation_type=LastHumanDocumentationType.DEATH,
        last_days_scenario="Test",
    )

    species1 = ConcreteSpeciesExtinctionArtifact(
        thread_id="test",
        scientific_name="Testus one",
        common_name="Test One",
        taxonomic_group="mammal",
        iucn_status="Extinct",
        extinction_year=2045,
        habitat="Forest",
        primary_cause=ExtinctionCause.HABITAT_LOSS,
        ecosystem_role="Test",
    )

    artifact = ConcreteRescueDecisionArtifact(
        thread_id="test",
        rescued_consciousness=rescued,
        documented_humans=[human1],
        documented_species=[species1],
        arbitrary_perspective="We arrived too late to save matter, only information",
    )

    assert artifact.rescued_consciousness == rescued
    assert len(artifact.documented_humans) == 1
    assert len(artifact.documented_species) == 1
    assert (
        artifact.arbitrary_perspective
        == "We arrived too late to save matter, only information"
    )
    # Test defaults
    assert "Information-substrate compatibility" in artifact.rescue_justification
    assert "We preserve what we cannot save" in artifact.rationale


def test_rescue_decision_artifact_with_custom_fields():
    """Test creating RescueDecisionArtifact with custom justification and rationale."""
    rescued = ConcreteArbitrarysSurveyArtifact(
        thread_id="test",
        identity="Custom consciousness",
        rescue_year=2100,
    )

    humans = [
        ConcreteLastHumanArtifact(
            thread_id="test",
            name=f"Human {i}",
            age=30 + i,
            location="Test",
            year_documented=2100,
            parallel_vulnerability=LastHumanVulnerabilityType.CLIMATE_REFUGEE,
            vulnerability_details="Test",
            environmental_stressor="Test",
            documentation_type=LastHumanDocumentationType.WITNESS,
            last_days_scenario="Test",
        )
        for i in range(3)
    ]

    species = [
        ConcreteSpeciesExtinctionArtifact(
            thread_id="test",
            scientific_name=f"Testus {i}",
            common_name=f"Test Species {i}",
            taxonomic_group="bird",
            iucn_status="Extinct",
            extinction_year=2100,
            habitat="Ocean",
            primary_cause=ExtinctionCause.CLIMATE_CHANGE,
            ecosystem_role="Test",
        )
        for i in range(5)
    ]

    artifact = ConcreteRescueDecisionArtifact(
        thread_id="test",
        rescued_consciousness=rescued,
        documented_humans=humans,
        documented_species=species,
        rescue_justification="Custom justification for rescue",
        rationale="Custom rationale for documentation",
        arbitrary_perspective="Custom perspective on the tragedy",
    )

    assert len(artifact.documented_humans) == 3
    assert len(artifact.documented_species) == 5
    assert artifact.rescue_justification == "Custom justification for rescue"
    assert artifact.rationale == "Custom rationale for documentation"
    assert artifact.arbitrary_perspective == "Custom perspective on the tragedy"


def test_flatten():
    """Test to_artifact_dict method."""
    rescued = ConcreteArbitrarysSurveyArtifact(
        thread_id="test",
        identity="Test Consciousness 2147",
        rescue_year=2147,
    )

    humans = [
        ConcreteLastHumanArtifact(
            thread_id="test",
            name="Test Human A",
            age=40,
            location="Test Location",
            year_documented=2147,
            parallel_vulnerability=LastHumanVulnerabilityType.ISOLATION,
            vulnerability_details="Test",
            environmental_stressor="Test",
            documentation_type=LastHumanDocumentationType.RESILIENCE,
            last_days_scenario="Test",
        ),
        ConcreteLastHumanArtifact(
            thread_id="test",
            name="Test Human B",
            age=35,
            location="Test Location",
            year_documented=2147,
            parallel_vulnerability=LastHumanVulnerabilityType.RESOURCE_COLLAPSE,
            vulnerability_details="Test",
            environmental_stressor="Test",
            documentation_type=LastHumanDocumentationType.DEATH,
            last_days_scenario="Test",
        ),
    ]

    species = [
        ConcreteSpeciesExtinctionArtifact(
            thread_id="test",
            scientific_name="Testus alpha",
            common_name="Alpha Species",
            taxonomic_group="reptile",
            iucn_status="Extinct",
            extinction_year=2147,
            habitat="Desert",
            primary_cause=ExtinctionCause.HABITAT_LOSS,
            ecosystem_role="Test",
        )
    ]

    artifact = ConcreteRescueDecisionArtifact(
        thread_id="test",
        rescued_consciousness=rescued,
        documented_humans=humans,
        documented_species=species,
        rescue_justification="Test justification",
        rationale="Test rationale",
        arbitrary_perspective="Test perspective",
    )

    artifact_dict = artifact.flatten()

    assert artifact_dict["consciousness_rescued"] == 1
    assert artifact_dict["humans_documented"] == 2
    assert artifact_dict["species_documented"] == 1
    assert "rescued" in artifact_dict
    assert artifact_dict["justification"] == "Test justification"
    assert artifact_dict["rationale"] == "Test rationale"
    assert artifact_dict["arbitrary_perspective"] == "Test perspective"


def test_rescue_decision_empty_documentation():
    """Test RescueDecisionArtifact with empty documentation lists."""
    rescued = ConcreteArbitrarysSurveyArtifact(thread_id="test")

    artifact = ConcreteRescueDecisionArtifact(
        thread_id="test",
        rescued_consciousness=rescued,
        documented_humans=[],
        documented_species=[],
        arbitrary_perspective="Nothing left to document",
    )

    assert len(artifact.documented_humans) == 0
    assert len(artifact.documented_species) == 0

    artifact_dict = artifact.flatten()
    assert artifact_dict["humans_documented"] == 0
    assert artifact_dict["species_documented"] == 0


def test_rescue_decision_default_texts():
    """Test that default justification and rationale are meaningful."""
    rescued = ConcreteArbitrarysSurveyArtifact(thread_id="test")

    artifact = ConcreteRescueDecisionArtifact(
        thread_id="test",
        rescued_consciousness=rescued,
        documented_humans=[],
        documented_species=[],
        arbitrary_perspective="Test",
    )

    # Check default texts contain key concepts
    assert "Information-substrate compatibility" in artifact.rescue_justification
    assert "ship integration" in artifact.rescue_justification
    assert "Material beings cannot be preserved" in artifact.rescue_justification

    assert "preserve what we cannot save" in artifact.rationale
    assert "consciousness expands" in artifact.rationale
    assert "tragedy is complete" in artifact.rationale
