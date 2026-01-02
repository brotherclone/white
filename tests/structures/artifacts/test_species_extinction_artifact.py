from pydantic import ValidationError
import pytest

from app.structures.artifacts.species_extinction_artifact import (
    SpeciesExtinctionArtifact,
)
from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.concepts.population_data import PopulationData
from app.structures.enums.extinction_cause import ExtinctionCause


class ConcreteSpeciesExtinctionArtifact(SpeciesExtinctionArtifact):
    """Concrete implementation for testing"""

    thread_id: str = "test-thread"

    def save_file(self):
        pass


def test_inheritance():
    assert issubclass(SpeciesExtinctionArtifact, ChainArtifact)


def test_species_extinction_artifact_required_fields():
    """Test creating SpeciesExtinctionArtifact with required fields only."""
    artifact = ConcreteSpeciesExtinctionArtifact(
        thread_id="test-thread",
        scientific_name="Rhinoceros sondaicus",
        common_name="Javan Rhino",
        taxonomic_group="mammal",
        iucn_status="Critically Endangered",
        extinction_year=2045,
        habitat="Tropical rainforest",
        primary_cause=ExtinctionCause.HABITAT_LOSS,
        ecosystem_role="Megaherbivore, habitat engineer",
    )

    assert artifact.scientific_name == "Rhinoceros sondaicus"
    assert artifact.common_name == "Javan Rhino"
    assert artifact.taxonomic_group == "mammal"
    assert artifact.iucn_status == "Critically Endangered"
    assert artifact.extinction_year == 2045
    assert artifact.habitat == "Tropical rainforest"
    assert artifact.primary_cause == ExtinctionCause.HABITAT_LOSS
    assert artifact.ecosystem_role == "Megaherbivore, habitat engineer"
    assert artifact.endemic is False
    assert artifact.range_km2 is None
    assert artifact.secondary_causes == []
    assert artifact.anthropogenic_factors == []
    assert artifact.cascade_effects == []
    assert artifact.symbolic_resonance == []
    assert artifact.human_parallel_hints == []
    assert artifact.narrative_potential_score == 0.0
    assert artifact.symbolic_weight == 0.0
    assert artifact.size_category == "medium"
    assert artifact.lifespan_years is None
    assert artifact.movement_pattern is None


def test_species_extinction_artifact_all_fields():
    """Test creating SpeciesExtinctionArtifact with all fields populated."""
    population_data = [
        PopulationData(year=2020, population=50000, source="census", confidence="high"),
        PopulationData(
            year=2030, population=10000, source="estimated", confidence="medium"
        ),
        PopulationData(year=2040, population=500, source="estimated", confidence="low"),
    ]

    artifact = ConcreteSpeciesExtinctionArtifact(
        thread_id="test-thread",
        scientific_name="Phocoena sinus",
        common_name="Vaquita",
        taxonomic_group="marine mammal",
        iucn_status="Critically Endangered",
        extinction_year=2030,
        population_trajectory=population_data,
        habitat="Gulf of California",
        endemic=True,
        range_km2=4000.0,
        primary_cause=ExtinctionCause.BYCATCH,
        secondary_causes=["habitat degradation", "illegal fishing"],
        anthropogenic_factors=["gillnet fishing", "illegal totoaba trade"],
        cascade_effects=["predator-prey imbalance", "ecosystem simplification"],
        ecosystem_role="Small cetacean, mid-level predator",
        symbolic_resonance=["Last of its kind", "Failed conservation"],
        human_parallel_hints=[
            "Fishing communities losing livelihoods",
            "Victims of industrial fishing",
        ],
        narrative_potential_score=0.95,
        symbolic_weight=0.9,
        size_category="small",
        lifespan_years=20,
        movement_pattern="sedentary",
    )

    assert artifact.common_name == "Vaquita"
    assert artifact.endemic is True
    assert artifact.range_km2 == 4000.0
    assert len(artifact.population_trajectory) == 3
    assert len(artifact.secondary_causes) == 2
    assert len(artifact.anthropogenic_factors) == 2
    assert artifact.narrative_potential_score == 0.95
    assert artifact.symbolic_weight == 0.9
    assert artifact.size_category == "small"
    assert artifact.lifespan_years == 20
    assert artifact.movement_pattern == "sedentary"


def test_extinction_year_validation():
    """Test extinction_year validation (2020-2150)."""
    # Valid years
    artifact_2020 = ConcreteSpeciesExtinctionArtifact(
        thread_id="test",
        scientific_name="Test species",
        common_name="Test",
        taxonomic_group="bird",
        iucn_status="Extinct",
        extinction_year=2020,
        habitat="Forest",
        primary_cause=ExtinctionCause.HABITAT_LOSS,
        ecosystem_role="Pollinator",
    )
    assert artifact_2020.extinction_year == 2020

    artifact_2150 = ConcreteSpeciesExtinctionArtifact(
        thread_id="test",
        scientific_name="Test species",
        common_name="Test",
        taxonomic_group="bird",
        iucn_status="Extinct",
        extinction_year=2150,
        habitat="Forest",
        primary_cause=ExtinctionCause.CLIMATE_CHANGE,
        ecosystem_role="Seed disperser",
    )
    assert artifact_2150.extinction_year == 2150

    # Invalid: too early
    with pytest.raises(ValidationError):
        ConcreteSpeciesExtinctionArtifact(
            thread_id="test",
            scientific_name="Test",
            common_name="Test",
            taxonomic_group="bird",
            iucn_status="Extinct",
            extinction_year=1919,
            habitat="Forest",
            primary_cause=ExtinctionCause.HABITAT_LOSS,
            ecosystem_role="Test",
        )

    # Invalid: too late
    with pytest.raises(ValidationError):
        ConcreteSpeciesExtinctionArtifact(
            thread_id="test",
            scientific_name="Test",
            common_name="Test",
            taxonomic_group="bird",
            iucn_status="Extinct",
            extinction_year=4444,
            habitat="Forest",
            primary_cause=ExtinctionCause.HABITAT_LOSS,
            ecosystem_role="Test",
        )


def test_narrative_potential_score_validation():
    """Test narrative_potential_score validation (0.0-1.0)."""
    # Valid scores
    artifact = ConcreteSpeciesExtinctionArtifact(
        thread_id="test",
        scientific_name="Test",
        common_name="Test",
        taxonomic_group="mammal",
        iucn_status="Endangered",
        extinction_year=2050,
        habitat="Ocean",
        primary_cause=ExtinctionCause.POLLUTION,
        ecosystem_role="Filter feeder",
        narrative_potential_score=0.5,
    )
    assert artifact.narrative_potential_score == 0.5

    # Invalid: negative
    with pytest.raises(ValidationError):
        ConcreteSpeciesExtinctionArtifact(
            thread_id="test",
            scientific_name="Test",
            common_name="Test",
            taxonomic_group="mammal",
            iucn_status="Endangered",
            extinction_year=2050,
            habitat="Ocean",
            primary_cause=ExtinctionCause.POLLUTION,
            ecosystem_role="Filter feeder",
            narrative_potential_score=-0.1,
        )

    # Invalid: greater than 1
    with pytest.raises(ValidationError):
        ConcreteSpeciesExtinctionArtifact(
            thread_id="test",
            scientific_name="Test",
            common_name="Test",
            taxonomic_group="mammal",
            iucn_status="Endangered",
            extinction_year=2050,
            habitat="Ocean",
            primary_cause=ExtinctionCause.POLLUTION,
            ecosystem_role="Filter feeder",
            narrative_potential_score=1.1,
        )


def test_size_category_literal():
    """Test that size_category only accepts valid literal values."""
    for size in ["tiny", "small", "medium", "large", "massive"]:
        artifact = ConcreteSpeciesExtinctionArtifact(
            thread_id="test",
            scientific_name="Test",
            common_name="Test",
            taxonomic_group="insect",
            iucn_status="Endangered",
            extinction_year=2050,
            habitat="Forest",
            primary_cause=ExtinctionCause.HABITAT_LOSS,
            ecosystem_role="Pollinator",
            size_category=size,
        )
        assert artifact.size_category == size

    # Invalid value
    with pytest.raises(ValidationError):
        ConcreteSpeciesExtinctionArtifact(
            thread_id="test",
            scientific_name="Test",
            common_name="Test",
            taxonomic_group="insect",
            iucn_status="Endangered",
            extinction_year=2050,
            habitat="Forest",
            primary_cause=ExtinctionCause.HABITAT_LOSS,
            ecosystem_role="Pollinator",
            size_category="gigantic",
        )


def test_flatten():
    """Test to_artifact_dict method."""
    population_data = [
        PopulationData(year=2025, population=1000),
        PopulationData(year=2035, population=100),
    ]

    artifact = ConcreteSpeciesExtinctionArtifact(
        thread_id="test",
        scientific_name="Testus extinctus",
        common_name="Test Species",
        taxonomic_group="bird",
        iucn_status="Critically Endangered",
        extinction_year=2040,
        habitat="Coastal wetlands",
        primary_cause=ExtinctionCause.CLIMATE_CHANGE,
        ecosystem_role="Wading bird",
        population_trajectory=population_data,
        cascade_effects=["Wetland degradation"],
        anthropogenic_factors=["Sea level rise"],
        symbolic_resonance=["Loss of coastal ecosystems"],
        narrative_potential_score=0.8,
        symbolic_weight=0.75,
    )

    artifact_dict = artifact.flatten()

    assert artifact_dict["species"] == "Test Species"
    assert artifact_dict["scientific_name"] == "Testus extinctus"
    assert artifact_dict["extinction_year"] == 2040
    assert artifact_dict["primary_cause"] == "climate_change"
    assert artifact_dict["habitat"] == "Coastal wetlands"
    assert len(artifact_dict["population_trajectory"]) == 2
    assert artifact_dict["cascade_effects"] == ["Wetland degradation"]
    assert artifact_dict["anthropogenic_factors"] == ["Sea level rise"]
    assert artifact_dict["symbolic_resonance"] == ["Loss of coastal ecosystems"]
    assert artifact_dict["narrative_score"] == 0.8
    assert artifact_dict["symbolic_weight"] == 0.75


def test_summary_text():
    """Test summary_text method."""
    population_data = [
        PopulationData(year=2020, population=5000),
        PopulationData(year=2040, population=0),
    ]

    artifact = ConcreteSpeciesExtinctionArtifact(
        thread_id="test",
        scientific_name="Testus summarius",
        common_name="Summary Species",
        taxonomic_group="reptile",
        iucn_status="Extinct",
        extinction_year=2040,
        habitat="Desert",
        primary_cause=ExtinctionCause.HABITAT_LOSS,
        ecosystem_role="Burrowing lizard",
        population_trajectory=population_data,
        cascade_effects=["Soil degradation", "Loss of prey base", "Predator decline"],
    )

    summary = artifact.summary_text()

    assert "Summary Species" in summary
    assert "Testus summarius" in summary
    assert "2040" in summary
    assert "habitat loss" in summary
    assert "Desert" in summary
    assert "5000" in summary
    assert "2020" in summary
    assert "Soil degradation" in summary


def test_summary_text_without_population():
    """Test summary_text when no population data exists."""
    artifact = ConcreteSpeciesExtinctionArtifact(
        thread_id="test",
        scientific_name="Testus nopopulus",
        common_name="No Pop Species",
        taxonomic_group="fish",
        iucn_status="Extinct",
        extinction_year=2050,
        habitat="River",
        primary_cause=ExtinctionCause.POLLUTION,
        ecosystem_role="Bottom feeder",
        cascade_effects=["Water quality decline"],
    )

    summary = artifact.summary_text()

    assert "No Pop Species" in summary
    assert "Testus nopopulus" in summary
    assert "2050" in summary
    assert "pollution" in summary
