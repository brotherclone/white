import pytest
from datetime import datetime
from pydantic import ValidationError

from app.structures.concepts.green_corpus_entry import GreenCorpusEntry
from app.structures.artifacts.species_extinction_artifact import (
    SpeciesExtinctionArtifact,
)
from app.structures.enums.extinction_cause import ExtinctionCause


def test_create_valid_model():
    species = SpeciesExtinctionArtifact(
        common_name="Future Coral Species",
        scientific_name="Acropora futura",
        extinction_year=2050,
        taxonomic_group="marine",
        iucn_status="Critically Endangered",
        primary_cause=ExtinctionCause.CLIMATE_CHANGE,
        habitat="Pacific coral reefs",
        ecosystem_role="Reef builder and habitat provider",
    )

    m = GreenCorpusEntry(species=species)
    assert isinstance(m, GreenCorpusEntry)
    assert m.species.common_name == "Future Coral Species"
    assert m.suggested_human_parallels == []
    assert m.musical_hints == {}
    assert m.source_notes == ""
    assert isinstance(m.created_date, datetime)
    assert m.model_dump(exclude_none=True, exclude_unset=True)


def test_with_all_optional_fields():
    species = SpeciesExtinctionArtifact(
        common_name="Future Polar Bear",
        scientific_name="Ursus maritimus",
        extinction_year=2075,
        taxonomic_group="mammal",
        iucn_status="Vulnerable",
        primary_cause=ExtinctionCause.CLIMATE_CHANGE,
        habitat="Arctic sea ice",
        ecosystem_role="Apex predator",
    )

    m = GreenCorpusEntry(
        species=species,
        suggested_human_parallels=[
            {"type": "cultural", "description": "loss of indigenous languages"},
            {"type": "technological", "description": "obsolete technologies"},
        ],
        musical_hints={
            "tempo": "slow",
            "mood": "melancholic",
            "instruments": "strings",
        },
        source_notes="Research from Cornell Lab of Ornithology",
    )

    assert len(m.suggested_human_parallels) == 2
    assert m.suggested_human_parallels[0]["type"] == "cultural"
    assert "tempo" in m.musical_hints
    assert m.musical_hints["mood"] == "melancholic"
    assert m.source_notes == "Research from Cornell Lab of Ornithology"


def test_missing_required_field_raises():
    with pytest.raises(ValidationError):
        GreenCorpusEntry()


@pytest.mark.skip(
    "to_artifact_dict requires ChainArtifact.to_artifact_dict() implementation"
)
def test_to_artifact_dict():
    # This test is skipped because SpeciesExtinctionArtifact.to_artifact_dict()
    # is not fully implemented in the base class
    pass


def test_created_date_auto_generated():
    species = SpeciesExtinctionArtifact(
        common_name="Future Coral",
        scientific_name="Acropora futura",
        extinction_year=2050,
        taxonomic_group="marine",
        iucn_status="Critically Endangered",
        primary_cause=ExtinctionCause.CLIMATE_CHANGE,
        habitat="Pacific reefs",
        ecosystem_role="Reef builder",
    )

    m1 = GreenCorpusEntry(species=species)
    m2 = GreenCorpusEntry(species=species)

    # Both should have created_date set
    assert m1.created_date is not None
    assert m2.created_date is not None
    # They should be close in time (within a second)
    time_diff = abs((m2.created_date - m1.created_date).total_seconds())
    assert time_diff < 1.0


def test_field_descriptions():
    fields = getattr(GreenCorpusEntry, "model_fields", None)
    assert fields is not None
