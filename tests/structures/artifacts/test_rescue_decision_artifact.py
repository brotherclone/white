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


def test_for_prompt_basic():
    """Test for_prompt() returns structured output."""
    rescued = ConcreteArbitrarysSurveyArtifact(
        thread_id="test",
        identity="Mind Instance 42",
        rescue_year=2100,
    )

    human = ConcreteLastHumanArtifact(
        thread_id="test",
        name="John Doe",
        age=45,
        location="Earth",
        year_documented=2100,
        parallel_vulnerability=LastHumanVulnerabilityType.DISPLACEMENT,
        vulnerability_details="Test",
        environmental_stressor="Test",
        documentation_type=LastHumanDocumentationType.DEATH,
        last_days_scenario="Test",
    )

    species = ConcreteSpeciesExtinctionArtifact(
        thread_id="test",
        scientific_name="Homo sapiens",
        common_name="Human",
        taxonomic_group="mammal",
        iucn_status="Extinct",
        extinction_year=2100,
        habitat="Global",
        primary_cause=ExtinctionCause.CLIMATE_CHANGE,
        ecosystem_role="Test",
    )

    artifact = ConcreteRescueDecisionArtifact(
        thread_id="test",
        rescued_consciousness=rescued,
        documented_humans=[human],
        documented_species=[species],
        arbitrary_perspective="We witnessed the end",
    )

    prompt = artifact.for_prompt()

    assert "## The Arbitrary's Decision" in prompt
    assert "Rescued: Mind Instance 42" in prompt
    assert "Documented (not rescued): 1 humans, 1 species" in prompt
    assert "## Justification" in prompt
    assert "## Rationale" in prompt
    assert "## The Mind's Reflection" in prompt
    assert "We witnessed the end" in prompt


def test_for_prompt_multiple_documentations():
    """Test for_prompt() with multiple humans and species."""
    rescued = ConcreteArbitrarysSurveyArtifact(
        thread_id="test",
        identity="Consciousness Alpha",
        rescue_year=2140,
    )

    humans = [
        ConcreteLastHumanArtifact(
            thread_id="test",
            name=f"Human {i}",
            age=30 + i,
            location="Test",
            year_documented=2140,
            parallel_vulnerability=LastHumanVulnerabilityType.CLIMATE_REFUGEE,
            vulnerability_details="Test",
            environmental_stressor="Test",
            documentation_type=LastHumanDocumentationType.WITNESS,
            last_days_scenario="Test",
        )
        for i in range(10)
    ]

    species = [
        ConcreteSpeciesExtinctionArtifact(
            thread_id="test",
            scientific_name=f"Species {i}",
            common_name=f"Test {i}",
            taxonomic_group="mammal",
            iucn_status="Extinct",
            extinction_year=2140,
            habitat="Test",
            primary_cause=ExtinctionCause.HABITAT_LOSS,
            ecosystem_role="Test",
        )
        for i in range(25)
    ]

    artifact = ConcreteRescueDecisionArtifact(
        thread_id="test",
        rescued_consciousness=rescued,
        documented_humans=humans,
        documented_species=species,
        arbitrary_perspective="The scale of loss defies comprehension",
    )

    prompt = artifact.for_prompt()

    assert "10 humans, 25 species" in prompt
    assert "The scale of loss defies comprehension" in prompt


def test_for_prompt_custom_justification():
    """Test for_prompt() with custom justification and rationale."""
    rescued = ConcreteArbitrarysSurveyArtifact(
        thread_id="test",
        identity="Test Mind",
        rescue_year=2150,
    )

    artifact = ConcreteRescueDecisionArtifact(
        thread_id="test",
        rescued_consciousness=rescued,
        documented_humans=[],
        documented_species=[],
        rescue_justification="This consciousness showed unique properties",
        rationale="All choices involve loss; we chose to preserve one spark",
        arbitrary_perspective="Even one saved is a defiance of entropy",
    )

    prompt = artifact.for_prompt()

    assert "This consciousness showed unique properties" in prompt
    assert "All choices involve loss" in prompt
    assert "defiance of entropy" in prompt


def test_for_prompt_zero_documentations():
    """Test for_prompt() with no documented humans or species."""
    rescued = ConcreteArbitrarysSurveyArtifact(
        thread_id="test",
        identity="Sole Survivor",
        rescue_year=2300,
    )

    artifact = ConcreteRescueDecisionArtifact(
        thread_id="test",
        rescued_consciousness=rescued,
        documented_humans=[],
        documented_species=[],
        arbitrary_perspective="Only information remains",
    )

    prompt = artifact.for_prompt()

    assert "0 humans, 0 species" in prompt
    assert "Only information remains" in prompt


def test_for_prompt_sections_order():
    """Test that for_prompt() maintains section order."""
    rescued = ConcreteArbitrarysSurveyArtifact(
        thread_id="test",
        identity="Test",
        rescue_year=2100,
    )

    artifact = ConcreteRescueDecisionArtifact(
        thread_id="test",
        rescued_consciousness=rescued,
        documented_humans=[],
        documented_species=[],
        arbitrary_perspective="Test",
    )

    prompt = artifact.for_prompt()
    lines = prompt.split("\n")

    # Find section headers
    decision_idx = next(
        i for i, l in enumerate(lines) if "## The Arbitrary's Decision" in l
    )
    justification_idx = next(i for i, l in enumerate(lines) if "## Justification" in l)
    rationale_idx = next(i for i, l in enumerate(lines) if "## Rationale" in l)
    reflection_idx = next(
        i for i, l in enumerate(lines) if "## The Mind's Reflection" in l
    )

    # Verify order
    assert decision_idx < justification_idx
    assert justification_idx < rationale_idx
    assert rationale_idx < reflection_idx


def test_for_prompt_with_all_fields():
    """Test for_prompt() includes all relevant information."""
    rescued = ConcreteArbitrarysSurveyArtifact(
        thread_id="test",
        identity="Complete Mind Instance",
        rescue_year=2145,
    )

    humans = [
        ConcreteLastHumanArtifact(
            thread_id="test",
            name="Last Human",
            age=50,
            location="Final City",
            year_documented=2145,
            parallel_vulnerability=LastHumanVulnerabilityType.ENTANGLEMENT,
            vulnerability_details="Caught in collapse",
            environmental_stressor="Total system failure",
            documentation_type=LastHumanDocumentationType.DEATH,
            last_days_scenario="Final moments",
        )
    ]

    species = [
        ConcreteSpeciesExtinctionArtifact(
            thread_id="test",
            scientific_name="Testus finalis",
            common_name="Final Species",
            taxonomic_group="bird",
            iucn_status="Extinct",
            extinction_year=2145,
            habitat="Sky",
            primary_cause=ExtinctionCause.POLLUTION,
            ecosystem_role="Indicator",
        )
    ]

    artifact = ConcreteRescueDecisionArtifact(
        thread_id="test",
        rescued_consciousness=rescued,
        documented_humans=humans,
        documented_species=species,
        rescue_justification="Complete test justification for rescue decision",
        rationale="Complete test rationale for documentation and preservation",
        arbitrary_perspective="Complete perspective on the tragedy and loss",
    )

    prompt = artifact.for_prompt()

    # Check all sections present
    assert "Complete Mind Instance" in prompt
    assert "1 humans, 1 species" in prompt
    assert "Complete test justification" in prompt
    assert "Complete test rationale" in prompt
    assert "Complete perspective" in prompt
