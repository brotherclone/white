from datetime import datetime
from pydantic import BaseModel

from app.structures.concepts.green_corpus_entry import GreenCorpusEntry
from app.structures.artifacts.species_extinction_artifact import (
    SpeciesExtinctionArtifact,
)
from app.structures.enums.extinction_cause import ExtinctionCause


class ConcreteSpeciesExtinctionArtifact(SpeciesExtinctionArtifact):
    """Concrete implementation for testing"""

    thread_id: str = "test-thread"

    def flatten(self):
        return self.model_dump()

    def save_file(self):
        pass

    def to_artifact_dict(self):
        return super().to_artifact_dict()


def test_inheritance():
    assert issubclass(GreenCorpusEntry, BaseModel)


def test_green_corpus_entry_creation_with_required_fields():
    """Test creating GreenCorpusEntry with required fields only."""
    species = ConcreteSpeciesExtinctionArtifact(
        thread_id="test-thread",
        scientific_name="Panthera tigris sumatrae",
        common_name="Sumatran Tiger",
        taxonomic_group="mammal",
        iucn_status="Critically Endangered",
        extinction_year=2050,
        habitat="Tropical rainforest",
        primary_cause=ExtinctionCause.HABITAT_LOSS,
        ecosystem_role="Apex predator",
    )

    entry = GreenCorpusEntry(species=species)

    assert entry.species == species
    assert entry.suggested_human_parallels == []
    assert entry.musical_hints == {}
    assert isinstance(entry.created_date, datetime)
    assert entry.source_notes == ""


def test_green_corpus_entry_creation_with_all_fields():
    """Test creating GreenCorpusEntry with all fields populated."""
    species = ConcreteSpeciesExtinctionArtifact(
        thread_id="test-thread",
        scientific_name="Phocoena sinus",
        common_name="Vaquita",
        taxonomic_group="marine mammal",
        iucn_status="Critically Endangered",
        extinction_year=2030,
        habitat="Gulf of California",
        primary_cause=ExtinctionCause.BYCATCH,
        ecosystem_role="Small cetacean",
    )

    parallels = [
        {
            "type": "displacement",
            "description": "Fishing communities losing livelihoods",
        },
        {"type": "entanglement", "description": "Victims of industrial fishing"},
    ]

    hints = {"tempo": "slow", "instrumentation": "cello, ocean sounds"}

    test_date = datetime(2024, 1, 1)
    notes = "Research from conservation biology papers"

    entry = GreenCorpusEntry(
        species=species,
        suggested_human_parallels=parallels,
        musical_hints=hints,
        created_date=test_date,
        source_notes=notes,
    )

    assert entry.species == species
    assert len(entry.suggested_human_parallels) == 2
    assert entry.musical_hints["tempo"] == "slow"
    assert entry.created_date == test_date
    assert entry.source_notes == notes


def test_green_corpus_entry_to_artifact_dict():
    """Test to_artifact_dict method."""
    species = ConcreteSpeciesExtinctionArtifact(
        thread_id="test-thread",
        scientific_name="Chelonia mydas",
        common_name="Green Sea Turtle",
        taxonomic_group="reptile",
        iucn_status="Endangered",
        extinction_year=2075,
        habitat="Tropical and subtropical oceans",
        primary_cause=ExtinctionCause.CLIMATE_CHANGE,
        ecosystem_role="Herbivore, ecosystem engineer",
    )

    parallels = [{"type": "climate_refugee", "description": "Coastal displacement"}]
    hints = {"key": "E minor"}

    entry = GreenCorpusEntry(
        species=species,
        suggested_human_parallels=parallels,
        musical_hints=hints,
        source_notes="IUCN Red List data",
    )

    artifact_dict = entry.to_artifact_dict()

    assert "species" in artifact_dict
    assert "suggested_parallels" in artifact_dict
    assert "musical_hints" in artifact_dict
    assert "source_notes" in artifact_dict
    assert artifact_dict["suggested_parallels"] == parallels
    assert artifact_dict["musical_hints"] == hints
    assert artifact_dict["source_notes"] == "IUCN Red List data"


def test_green_corpus_entry_default_factory_independence():
    """Test that default factory creates independent lists/dicts for each instance."""
    species1 = ConcreteSpeciesExtinctionArtifact(
        thread_id="test-thread-1",
        scientific_name="Species One",
        common_name="One",
        taxonomic_group="bird",
        iucn_status="Endangered",
        extinction_year=2040,
        habitat="Forest",
        primary_cause=ExtinctionCause.HABITAT_LOSS,
        ecosystem_role="Pollinator",
    )

    species2 = ConcreteSpeciesExtinctionArtifact(
        thread_id="test-thread-2",
        scientific_name="Species Two",
        common_name="Two",
        taxonomic_group="mammal",
        iucn_status="Vulnerable",
        extinction_year=2060,
        habitat="Grassland",
        primary_cause=ExtinctionCause.CLIMATE_CHANGE,
        ecosystem_role="Grazer",
    )

    entry1 = GreenCorpusEntry(species=species1)
    entry2 = GreenCorpusEntry(species=species2)

    # Modify entry1's lists/dicts
    entry1.suggested_human_parallels.append({"type": "test"})
    entry1.musical_hints["test"] = "value"

    # Verify entry2 is not affected
    assert len(entry2.suggested_human_parallels) == 0
    assert len(entry2.musical_hints) == 0
